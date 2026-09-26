# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Tests for deadline_worker_agent.worker_protocol_trace and its hooks."""

from __future__ import annotations

import io
import json
import re
from datetime import timedelta
from threading import Event, RLock
from typing import Any, Generator, cast
from unittest.mock import MagicMock

import pytest

from openjd.sessions import ActionState, ActionStatus

import deadline_worker_agent.worker_protocol_trace as trace_mod
from deadline_worker_agent.api_models import AssignedSession, EnvironmentAction
from deadline_worker_agent.scheduler.scheduler import WorkerScheduler
from deadline_worker_agent.scheduler.session_action_status import SessionActionStatus
from deadline_worker_agent.worker import Worker
from deadline_worker_agent.worker_protocol_trace import (
    PROTOCOL_TRACE_ENV_VAR,
    PROTOCOL_TRACE_STRICT_ENV_VAR,
    TraceEvent,
    WorkerProtocolTrace,
    initialize_from_environment,
    scrub_protocol_trace_env_vars,
    set_trace,
)


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def trace(stream: io.StringIO) -> Generator[WorkerProtocolTrace, None, None]:
    """Installs an enabled, strict, stream-backed trace as the process-wide
    instance and restores the previous instance afterwards."""
    trace = WorkerProtocolTrace(enabled=True, stream=stream, strict=True)
    previous = set_trace(trace)
    yield trace
    set_trace(previous)


def _assert_canonical_v1(record: dict) -> None:
    """Assert that a single emitted record is a canonical v1 envelope."""
    assert record["v"] == 1
    assert record["src"] == "worker-agent"
    assert isinstance(record["run"], str) and record["run"]
    assert isinstance(record["seq"], int) and record["seq"] >= 1
    assert record["corr"] is None or isinstance(record["corr"], str)
    assert isinstance(record["payload"], dict)


def _records(stream: io.StringIO) -> list[dict]:
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    # Every emitted record must be a canonical v1 envelope.
    for record in records:
        _assert_canonical_v1(record)
    return records


class TestEnvelope:
    def test_canonical_envelope_version_is_1(self) -> None:
        # THEN - the emitted envelope format is canonical v1
        assert trace_mod.ENVELOPE_VERSION == 1

    def test_envelope_v1_fields(self, trace: WorkerProtocolTrace, stream: io.StringIO) -> None:
        # WHEN
        trace.emit(TraceEvent.SESSION_START, corr="session-123", queue_id="queue-1", job_id="job-1")

        # THEN
        (record,) = _records(stream)
        assert set(record.keys()) == {"v", "ts", "seq", "src", "event", "run", "corr", "payload"}
        assert record["v"] == 1
        assert record["seq"] == 1
        assert record["src"] == "worker-agent"
        assert record["event"] == "session.start"
        assert record["run"] == trace.run_id
        assert record["corr"] == "session-123"
        assert record["payload"] == {"queue_id": "queue-1", "job_id": "job-1"}
        # ISO-8601 UTC timestamp
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00$", record["ts"])

    def test_seq_increments(self, trace: WorkerProtocolTrace, stream: io.StringIO) -> None:
        # WHEN
        trace.emit(TraceEvent.PROCESS_START)
        trace.emit(TraceEvent.PROCESS_STOP)

        # THEN
        records = _records(stream)
        assert [r["seq"] for r in records] == [1, 2]
        assert len({r["run"] for r in records}) == 1

    def test_corr_defaults_to_none(self, trace: WorkerProtocolTrace, stream: io.StringIO) -> None:
        # WHEN
        trace.emit(TraceEvent.PROCESS_STOP)

        # THEN
        (record,) = _records(stream)
        assert record["corr"] is None
        assert record["payload"] == {}


class TestAllowlisting:
    def test_unknown_event_dropped_when_not_strict(self, stream: io.StringIO) -> None:
        # GIVEN
        trace = WorkerProtocolTrace(enabled=True, stream=stream, strict=False)

        # WHEN
        trace.emit("not.a.real.event")

        # THEN
        assert stream.getvalue() == ""

    def test_unknown_event_raises_when_strict(self, trace: WorkerProtocolTrace) -> None:
        # THEN
        with pytest.raises(ValueError, match="Unknown protocol trace event"):
            trace.emit("not.a.real.event")

    def test_disallowed_payload_key_dropped_when_not_strict(self, stream: io.StringIO) -> None:
        # GIVEN
        trace = WorkerProtocolTrace(enabled=True, stream=stream, strict=False)

        # WHEN
        trace.emit(TraceEvent.SESSION_START, queue_id="queue-1", secret="do-not-emit")

        # THEN
        (record,) = _records(stream)
        assert record["payload"] == {"queue_id": "queue-1"}

    def test_disallowed_payload_key_raises_when_strict(self, trace: WorkerProtocolTrace) -> None:
        # THEN
        with pytest.raises(ValueError, match="not allowlisted"):
            trace.emit(TraceEvent.SESSION_START, secret="do-not-emit")

    def test_non_scalar_payload_value_dropped_when_not_strict(self, stream: io.StringIO) -> None:
        # GIVEN
        trace = WorkerProtocolTrace(enabled=True, stream=stream, strict=False)

        # WHEN
        trace.emit(
            TraceEvent.SESSION_START,
            queue_id=cast(Any, {"nested": "dict"}),
            job_id="job-1",
        )

        # THEN
        (record,) = _records(stream)
        assert record["payload"] == {"job_id": "job-1"}

    def test_non_scalar_payload_value_raises_when_strict(self, trace: WorkerProtocolTrace) -> None:
        # THEN
        with pytest.raises(TypeError, match="must be a scalar"):
            trace.emit(TraceEvent.SESSION_START, queue_id=cast(Any, {"nested": "dict"}))

    def test_long_string_values_truncated(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # WHEN
        trace.emit(TraceEvent.SESSION_START, queue_id="q" * 2000)

        # THEN
        (record,) = _records(stream)
        assert len(record["payload"]["queue_id"]) == 512

    def test_all_events_have_payload_allowlists(self) -> None:
        # THEN - every allowlisted event has a payload key allowlist entry
        for event in TraceEvent:
            assert event.value in trace_mod._EVENT_PAYLOAD_KEYS


class TestDisabledAndFailures:
    def test_disabled_is_noop(self, stream: io.StringIO) -> None:
        # GIVEN
        trace = WorkerProtocolTrace(enabled=False, stream=stream, strict=True)

        # WHEN
        trace.emit(TraceEvent.PROCESS_START)
        trace.emit("not.a.real.event")  # not even validated when disabled

        # THEN
        assert stream.getvalue() == ""

    def test_default_process_trace_is_disabled(self) -> None:
        # GIVEN a freshly-constructed default instance
        trace = WorkerProtocolTrace()

        # THEN
        assert not trace.enabled

    def test_write_failure_swallowed_when_not_strict(self) -> None:
        # GIVEN
        broken_stream = MagicMock()
        broken_stream.write.side_effect = OSError("boom")
        trace = WorkerProtocolTrace(enabled=True, stream=broken_stream, strict=False)

        # WHEN / THEN - does not raise
        trace.emit(TraceEvent.PROCESS_START)

    def test_write_failure_raises_when_strict(self) -> None:
        # GIVEN
        broken_stream = MagicMock()
        broken_stream.write.side_effect = OSError("boom")
        trace = WorkerProtocolTrace(enabled=True, stream=broken_stream, strict=True)

        # THEN
        with pytest.raises(OSError):
            trace.emit(TraceEvent.PROCESS_START)

    def test_self_disables_after_repeated_write_failures(self) -> None:
        # GIVEN
        broken_stream = MagicMock()
        broken_stream.write.side_effect = OSError("boom")
        trace = WorkerProtocolTrace(enabled=True, stream=broken_stream, strict=False)

        # WHEN
        for _ in range(trace_mod._MAX_CONSECUTIVE_WRITE_FAILURES):
            trace.emit(TraceEvent.PROCESS_START)

        # THEN
        assert not trace.enabled

    def test_close_disables_and_never_raises(self, stream: io.StringIO) -> None:
        # GIVEN
        trace = WorkerProtocolTrace(enabled=True, stream=stream)

        # WHEN
        trace.close()
        trace.emit(TraceEvent.PROCESS_START)

        # THEN
        assert not trace.enabled
        assert stream.getvalue() == ""


class TestEnvironmentSettings:
    @pytest.fixture(autouse=True)
    def restore_trace(self) -> Generator[None, None, None]:
        previous = trace_mod.get_trace()
        yield
        set_trace(previous)

    def test_default_off_when_unset(self) -> None:
        # GIVEN
        environ: dict[str, str] = {}

        # WHEN
        trace = initialize_from_environment(environ)

        # THEN
        assert not trace.enabled
        assert not trace.strict

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "off", "no"])
    def test_disabled_values(self, value: str) -> None:
        # GIVEN
        environ = {PROTOCOL_TRACE_ENV_VAR: value}

        # WHEN
        trace = initialize_from_environment(environ)

        # THEN
        assert not trace.enabled

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", "stderr"])
    def test_enabled_stderr_values(self, value: str) -> None:
        # GIVEN
        environ = {PROTOCOL_TRACE_ENV_VAR: value}

        # WHEN
        trace = initialize_from_environment(environ)

        # THEN
        assert trace.enabled
        assert trace._file_path is None

    def test_other_value_is_file_path(self, tmp_path) -> None:
        # GIVEN
        trace_file = tmp_path / "trace.jsonl"
        environ = {PROTOCOL_TRACE_ENV_VAR: str(trace_file)}

        # WHEN
        trace = initialize_from_environment(environ)
        trace.emit(TraceEvent.PROCESS_START, platform="linux")
        trace.close()

        # THEN
        (record,) = [json.loads(line) for line in trace_file.read_text().splitlines()]
        _assert_canonical_v1(record)
        assert record["event"] == "process.start"
        assert record["payload"] == {"platform": "linux"}

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1", True), ("true", True), ("on", True), ("yes", True), ("0", False), ("", False)],
    )
    def test_strict_values(self, value: str, expected: bool) -> None:
        # GIVEN
        environ = {
            PROTOCOL_TRACE_ENV_VAR: "1",
            PROTOCOL_TRACE_STRICT_ENV_VAR: value,
        }

        # WHEN
        trace = initialize_from_environment(environ)

        # THEN
        assert trace.strict == expected

    def test_initialize_scrubs_env_vars(self) -> None:
        # GIVEN
        environ = {
            PROTOCOL_TRACE_ENV_VAR: "1",
            PROTOCOL_TRACE_STRICT_ENV_VAR: "1",
            "OTHER_VAR": "untouched",
        }

        # WHEN
        initialize_from_environment(environ)

        # THEN
        assert PROTOCOL_TRACE_ENV_VAR not in environ
        assert PROTOCOL_TRACE_STRICT_ENV_VAR not in environ
        assert environ["OTHER_VAR"] == "untouched"

    def test_initialize_installs_process_wide_trace(self) -> None:
        # WHEN
        trace = initialize_from_environment({PROTOCOL_TRACE_ENV_VAR: "1"})

        # THEN
        assert trace_mod.get_trace() is trace


class TestScrubEnvVars:
    def test_scrub_removes_trace_vars_only(self) -> None:
        # GIVEN
        env = {
            PROTOCOL_TRACE_ENV_VAR: "1",
            PROTOCOL_TRACE_STRICT_ENV_VAR: "1",
            "DEADLINE_SESSION_ID": "session-123",
            "AWS_PROFILE": "queue-profile",
        }

        # WHEN
        scrub_protocol_trace_env_vars(env)

        # THEN
        assert env == {
            "DEADLINE_SESSION_ID": "session-123",
            "AWS_PROFILE": "queue-profile",
        }

    def test_scrub_tolerates_absent_vars(self) -> None:
        # GIVEN
        env: dict[str, str] = {}

        # WHEN / THEN - does not raise
        scrub_protocol_trace_env_vars(env)
        assert env == {}


class TestWorkerHooks:
    def test_worker_run_emits_start_and_stop(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        worker = MagicMock(spec=Worker)
        worker._worker_id = "worker-123"

        # WHEN
        Worker.run(cast(Worker, worker))

        # THEN
        worker._run.assert_called_once_with()
        records = _records(stream)
        assert [r["event"] for r in records] == ["worker.start", "worker.stop"]
        assert records[0]["payload"] == {"worker_id": "worker-123"}

    def test_worker_run_emits_stop_on_error(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        worker = MagicMock(spec=Worker)
        worker._worker_id = "worker-123"
        worker._run.side_effect = RuntimeError("boom")

        # WHEN
        with pytest.raises(RuntimeError):
            Worker.run(cast(Worker, worker))

        # THEN
        events = [r["event"] for r in _records(stream)]
        assert events == ["worker.start", "worker.stop"]


class TestSchedulerHooks:
    def test_shutdown_emits_drain_requested(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._shutdown = Event()
        scheduler._wakeup = Event()

        # WHEN
        WorkerScheduler.shutdown(
            cast(WorkerScheduler, scheduler),
            fail_message="drain now",
            grace_time=timedelta(seconds=3),
        )

        # THEN
        (record,) = _records(stream)
        assert record["event"] == "drain.requested"
        assert record["payload"] == {"grace_seconds": 3.0}

    def test_shutdown_emits_drain_requested_without_grace(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._shutdown = Event()
        scheduler._wakeup = Event()

        # WHEN
        WorkerScheduler.shutdown(cast(WorkerScheduler, scheduler))

        # THEN
        (record,) = _records(stream)
        assert record["payload"] == {"grace_seconds": None}

    def test_handle_session_action_update_emits_action_complete(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._action_update_lock = RLock()
        scheduler._action_updates_map = {}
        scheduler._sessions = {}
        scheduler._wakeup = Event()
        action_status = SessionActionStatus(
            id="sessionaction-abc123",
            completed_status="SUCCEEDED",
            status=ActionStatus(state=ActionState.SUCCESS),
            session_id="session-xyz",
            kind="TaskRun",
        )

        # WHEN
        WorkerScheduler._handle_session_action_update(
            cast(WorkerScheduler, scheduler), action_status
        )

        # THEN
        (record,) = _records(stream)
        assert record["event"] == "action.complete"
        assert record["corr"] == "sessionaction-abc123"
        assert record["payload"] == {
            "status": "SUCCEEDED",
            "session_id": "session-xyz",
            "kind": "TaskRun",
        }

    def test_handle_session_action_update_no_event_when_not_completed(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._action_update_lock = RLock()
        scheduler._action_updates_map = {}
        scheduler._sessions = {}
        scheduler._wakeup = Event()
        action_status = SessionActionStatus(
            id="sessionaction-abc123",
            status=ActionStatus(state=ActionState.RUNNING, progress=50.0),
        )

        # WHEN
        WorkerScheduler._handle_session_action_update(
            cast(WorkerScheduler, scheduler), action_status
        )

        # THEN
        assert stream.getvalue() == ""

    def test_fail_all_actions_emits_session_fail_and_settles(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._action_updates_map = {}
        scheduler._wakeup = Event()
        assigned_session = AssignedSession(
            queueId="queue-1",
            jobId="job-1",
            sessionActions=[
                cast(
                    EnvironmentAction,
                    {
                        "sessionActionId": "sessionaction-1",
                        "actionType": "ENV_ENTER",
                        "environment": {"environmentId": "env-1"},
                    },
                ),
                cast(
                    EnvironmentAction,
                    {
                        "sessionActionId": "sessionaction-2",
                        "actionType": "ENV_EXIT",
                        "environment": {"environmentId": "env-1"},
                    },
                ),
            ],
        )

        # WHEN
        WorkerScheduler._fail_all_actions(
            cast(WorkerScheduler, scheduler),
            assigned_session,
            "some failure",
            session_id="session-77",
        )

        # THEN - the session failure and one local settle per action
        records = _records(stream)
        assert [r["event"] for r in records] == [
            "session.fail",
            "action.complete",
            "action.complete",
        ]
        assert records[0]["corr"] == "session-77"
        assert records[0]["payload"] == {"action_count": 2}
        assert records[1]["corr"] == "sessionaction-1"
        assert records[1]["payload"] == {
            "status": "FAILED",
            "session_id": "session-77",
            "kind": "ENV_ENTER",
        }
        assert records[2]["corr"] == "sessionaction-2"
        assert records[2]["payload"] == {
            "status": "NEVER_ATTEMPTED",
            "session_id": "session-77",
            "kind": "ENV_EXIT",
        }
        # AND the queued statuses carry the same trace metadata
        assert scheduler._action_updates_map["sessionaction-1"].session_id == "session-77"
        assert scheduler._action_updates_map["sessionaction-1"].kind == "ENV_ENTER"

    def test_trace_new_assignments_emits_each_action_once(
        self, trace: WorkerProtocolTrace, stream: io.StringIO
    ) -> None:
        # GIVEN
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._worker_id = "worker-123"
        scheduler._trace_seen_assigned_action_ids = set()
        assigned_sessions = {
            "session-1": cast(
                AssignedSession,
                {
                    "queueId": "queue-1",
                    "jobId": "job-1",
                    "sessionActions": [
                        {
                            "sessionActionId": "sessionaction-1",
                            "actionType": "ENV_ENTER",
                            "environmentId": "env-1",
                        },
                        {
                            "sessionActionId": "sessionaction-2",
                            "actionType": "TASK_RUN",
                        },
                    ],
                },
            ),
        }

        # WHEN - the same response content is applied twice
        WorkerScheduler._trace_new_assignments(
            cast(WorkerScheduler, scheduler), assigned_sessions=assigned_sessions
        )
        WorkerScheduler._trace_new_assignments(
            cast(WorkerScheduler, scheduler), assigned_sessions=assigned_sessions
        )

        # THEN - each action announced exactly once
        records = _records(stream)
        assert [r["event"] for r in records] == ["action.assigned", "action.assigned"]
        assert records[0]["corr"] == "sessionaction-1"
        assert records[0]["payload"] == {
            "session_id": "session-1",
            "kind": "ENV_ENTER",
            "env_id": "env-1",
            "worker_id": "worker-123",
        }
        assert records[1]["corr"] == "sessionaction-2"
        assert records[1]["payload"] == {
            "session_id": "session-1",
            "kind": "TASK_RUN",
            "env_id": None,
            "worker_id": "worker-123",
        }

    @pytest.mark.parametrize(
        "emit_failure_reason, state, expected_reason",
        [
            pytest.param(True, ActionState.TIMEOUT, "ACTION_TIMEOUT", id="timeout-supported"),
            pytest.param(False, ActionState.TIMEOUT, None, id="timeout-unsupported-model"),
            pytest.param(True, ActionState.FAILED, None, id="ordinary-failure"),
        ],
    )
    def test_sync_action_report_carries_wire_failure_reason_only(
        self,
        trace: WorkerProtocolTrace,
        stream: io.StringIO,
        emit_failure_reason: bool,
        state: ActionState,
        expected_reason: str | None,
    ) -> None:
        """action.report traces the closed failureReason exactly as sent on
        the wire (present only when the service model accepts it and the
        action timed out) and never the free-form progressMessage."""
        # GIVEN a scheduler with one FAILED report pending
        scheduler = MagicMock(spec=WorkerScheduler)
        scheduler._emit_failure_reason = emit_failure_reason
        scheduler._action_update_lock = RLock()
        scheduler._action_updates_map = {
            "sessionaction-1": SessionActionStatus(
                id="sessionaction-1",
                session_id="session-1",
                kind="TaskRun",
                status=ActionStatus(
                    state=state,
                    exit_code=-1,
                    fail_message="TIMEOUT - Exceeded the allotted runtime limit.",
                ),
                completed_status="FAILED",
            )
        }
        scheduler._updated_session_actions = (  # type: ignore[method-assign]
            lambda: WorkerScheduler._updated_session_actions(cast(WorkerScheduler, scheduler))
        )
        scheduler._updated_action_to_boto = (  # type: ignore[method-assign]
            lambda action: WorkerScheduler._updated_action_to_boto(
                cast(WorkerScheduler, scheduler), action
            )
        )
        scheduler._session_action_failure_reason = WorkerScheduler._session_action_failure_reason
        scheduler._deadline = MagicMock()
        scheduler._farm_id = "farm-1"
        scheduler._fleet_id = "fleet-1"
        scheduler._worker_id = "worker-1"
        scheduler._shutdown = Event()
        sent: dict[str, Any] = {}

        def fake_uws(**kwargs: Any) -> dict[str, Any]:
            sent.update(kwargs["updated_session_actions"])
            return {"assignedSessions": {}, "cancelSessionActions": {}, "updateIntervalSeconds": 5}

        # WHEN
        with pytest.MonkeyPatch.context() as mp:
            import deadline_worker_agent.scheduler.scheduler as scheduler_mod

            mp.setattr(scheduler_mod, "update_worker_schedule", fake_uws)
            WorkerScheduler._sync(cast(WorkerScheduler, scheduler), interruptable=True)

        # THEN the wire and the trace agree
        assert sent["sessionaction-1"].get("failureReason") == expected_reason
        assert "TIMEOUT" in sent["sessionaction-1"]["progressMessage"]
        (report,) = [r for r in _records(stream) if r["event"] == "action.report"]
        assert report["corr"] == "sessionaction-1"
        assert report["payload"]["status"] == "FAILED"
        assert report["payload"]["failure_reason"] == expected_reason
        assert report["payload"]["session_id"] == "session-1"
        # Never the free-form message.
        assert "progress_message" not in report["payload"]
        assert "Exceeded the allotted runtime limit" not in json.dumps(report["payload"])
        assert set(report["payload"]) <= trace_mod._EVENT_PAYLOAD_KEYS["action.report"]

    def test_trace_new_assignments_noop_when_disabled(self, stream: io.StringIO) -> None:
        # GIVEN a disabled trace
        previous = set_trace(WorkerProtocolTrace(enabled=False, stream=stream, strict=True))
        try:
            scheduler = MagicMock(spec=WorkerScheduler)
            scheduler._worker_id = "worker-123"
            scheduler._trace_seen_assigned_action_ids = set()
            assigned_sessions = {
                "session-1": cast(
                    AssignedSession,
                    {
                        "sessionActions": [
                            {"sessionActionId": "sessionaction-1", "actionType": "TASK_RUN"}
                        ],
                    },
                ),
            }

            # WHEN
            WorkerScheduler._trace_new_assignments(
                cast(WorkerScheduler, scheduler), assigned_sessions=assigned_sessions
            )
        finally:
            set_trace(previous)

        # THEN - nothing emitted and no memory growth
        assert stream.getvalue() == ""
        assert scheduler._trace_seen_assigned_action_ids == set()


class TestEntrypointHooks:
    def test_entrypoint_initializes_scrubs_and_emits_process_events(
        self, tmp_path, monkeypatch
    ) -> None:
        # GIVEN
        from deadline_worker_agent.config import ConfigurationError
        import deadline_worker_agent.startup.entrypoint as entrypoint_mod

        trace_file = tmp_path / "trace.jsonl"
        monkeypatch.setenv(PROTOCOL_TRACE_ENV_VAR, str(trace_file))
        monkeypatch.setenv(PROTOCOL_TRACE_STRICT_ENV_VAR, "1")
        config_cls = MagicMock()
        config_cls.load.side_effect = ConfigurationError("test-induced failure")
        monkeypatch.setattr(entrypoint_mod, "Configuration", config_cls)
        previous = trace_mod.get_trace()

        # WHEN
        with pytest.raises(SystemExit):
            entrypoint_mod.entrypoint(cli_args=[])

        # THEN
        set_trace(previous)
        import os

        assert PROTOCOL_TRACE_ENV_VAR not in os.environ
        assert PROTOCOL_TRACE_STRICT_ENV_VAR not in os.environ
        records = [json.loads(line) for line in trace_file.read_text().splitlines()]
        for record in records:
            _assert_canonical_v1(record)
        events = [r["event"] for r in records]
        assert events == ["process.start", "process.stop"]
        assert records[0]["payload"].keys() == {"agent_version", "platform"}
