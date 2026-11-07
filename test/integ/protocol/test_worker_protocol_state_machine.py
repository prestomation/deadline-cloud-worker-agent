# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Property-based test of the worker protocol.

Hypothesis plays the service. Each step is something the service can do: assign a new
session, add tasks, cancel an action, close a session, return a retryable error, drain
the worker, or just let time pass. The real WorkerScheduler runs against the hermetic
mock service in ``deadline_test_fixtures.deadline_mock``. After every step, the mock
service's protocol checker must have no violations. At the end, every session must be
complete: each action reported, and each entered environment exited.

When a run fails, Hypothesis shrinks it to a short sequence of steps and prints it.

Environment variables:

* ``WORKER_PROTOCOL_EXAMPLES``: number of runs (default 10).
* ``WORKER_PROTOCOL_STEPS``: maximum steps in one run (default 10).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from deadline_worker_agent.api_models import WorkerStatus

from .harness import (
    JOB_ID,
    assert_no_violations,
    assert_service_drain_stops,
    env_enter,
    start_agent,
    task_run,
)

pytestmark = pytest.mark.timeout(3600)

KNOWN_VIOLATIONS = frozenset({"exit-without-enter"})
"""Rules the agent is known to break. Each one has a strict xfail test in
test_worker_protocol_examples.py, so a fix shows up there. Ignoring them here lets the
state machine keep looking for new problems."""

_TASKS = st.builds(
    task_run,
    sleep=st.sampled_from(["0", "0.3", "2"]),
    exit_code=st.sampled_from([0, 0, 0, 1]),
)
_ENVIRONMENTS = st.lists(st.sampled_from(["Good", "BadEnter", "BadExit"]), unique=True, max_size=2)
_ERROR_COUNT = st.integers(min_value=1, max_value=2)
_RETRYABLE_ERRORS: st.SearchStrategy[tuple[str, str, dict[str, str], int]] = st.one_of(
    st.tuples(
        st.sampled_from(["UpdateWorkerSchedule", "BatchGetJobEntity"]),
        st.just("ThrottlingException"),
        st.just({}),
        _ERROR_COUNT,
    ),
    st.tuples(
        st.just("UpdateWorkerSchedule"),
        st.just("ConflictException"),
        st.just({"reason": "CONCURRENT_MODIFICATION", "resourceType": "worker"}),
        _ERROR_COUNT,
    ),
    # Only one: botocore retries one 5xx itself. Two reach the agent, which then exits
    # (see test_internal_server_error_is_retried in test_worker_protocol_examples.py).
    st.tuples(
        st.just("UpdateWorkerSchedule"),
        st.just("InternalServerErrorException"),
        st.just({}),
        st.just(1),
    ),
)


class WorkerProtocolMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="worker-protocol-"))
        self.monkeypatch = pytest.MonkeyPatch()
        self.agent, self._stop_agent = start_agent(self.tmp_path, self.monkeypatch)
        self.session_ids: list[str] = []
        self.stopped = False
        self.worker_drained = False

    # ---- helpers -------------------------------------------------------

    @property
    def service(self):  # type: ignore[no-untyped-def]
        return self.agent.service

    def _open_sessions(self) -> list[str]:
        return [
            session_id
            for session_id in self.session_ids
            if not self.service.sessions[session_id].ended
        ]

    def _sessions_accepting_work(self) -> list[str]:
        return [
            session_id
            for session_id in self._open_sessions()
            if self.service.sessions[session_id].accepts_work
            and self.service.sessions[session_id].blocked_at is None
        ]

    def _cancelable_actions(self) -> list[str]:
        return [
            action.session_action_id
            for session_id in self._open_sessions()
            for action in self.service.sessions[session_id].actions
            if not action.is_terminal and not action.canceled_by_service
        ]

    # ---- rules: what the service does ----------------------------------

    @precondition(lambda self: not self.stopped and len(self._open_sessions()) < 2)
    @rule(environments=_ENVIRONMENTS, tasks=st.lists(_TASKS, max_size=3))
    def assign_session(self, environments: list[str], tasks: list) -> None:
        actions = [env_enter(name) for name in environments] + tasks
        self.session_ids.append(
            self.service.open_session(
                worker_id=self.agent.worker_id,
                queue_id=self.agent.queue_id,
                job_id=JOB_ID,
                actions=actions,
            )
        )

    @precondition(lambda self: not self.stopped and self._sessions_accepting_work())
    @rule(data=st.data(), tasks=st.lists(_TASKS, min_size=1, max_size=2))
    def add_tasks(self, data: st.DataObject, tasks: list) -> None:
        session_id = data.draw(st.sampled_from(self._sessions_accepting_work()))
        try:
            self.service.add_actions(session_id, tasks)
        except ValueError:
            # A task failed after the session was picked; the service may not add tasks now.
            pass

    @precondition(lambda self: not self.stopped and self._cancelable_actions())
    @rule(data=st.data())
    def cancel_action(self, data: st.DataObject) -> None:
        self.service.cancel_action(data.draw(st.sampled_from(self._cancelable_actions())))

    @precondition(lambda self: not self.stopped and self._open_sessions())
    @rule(data=st.data())
    def close_session(self, data: st.DataObject) -> None:
        self.service.close_session(data.draw(st.sampled_from(self._open_sessions())))

    @precondition(lambda self: not self.stopped)
    @rule(error=_RETRYABLE_ERRORS)
    def return_retryable_error(self, error: tuple) -> None:
        operation, code, fields, count = error
        pending = self.agent.backend.injected_errors.get(operation, [])
        if code == "InternalServerErrorException" and pending:
            # Two 5xx in a row reach the agent, which exits on them (a known bug, see
            # test_internal_server_error_is_retried). Keep at most one queued.
            return
        self.agent.backend.inject_error(
            operation, code, count=count, resourceId=self.agent.worker_id, **fields
        )

    @precondition(lambda self: not self.stopped)
    @rule(syncs=st.integers(min_value=1, max_value=3))
    def let_time_pass(self, syncs: int) -> None:
        self.agent.wait_for_syncs(syncs, timeout=30)

    @precondition(lambda self: not self.stopped)
    @rule(immediate=st.booleans())
    def service_drains_worker(self, immediate: bool) -> None:
        assert_service_drain_stops(self.agent, immediate=immediate)
        self.stopped = True

    @precondition(lambda self: not self.stopped)
    @rule()
    def worker_drains_itself(self) -> None:
        self.agent.scheduler.shutdown(
            grace_time=timedelta(seconds=5), fail_message="The host is shutting down."
        )
        assert self.agent.join(timeout=60) is None
        self.agent.set_status(WorkerStatus.STOPPED)
        self.stopped = True
        self.worker_drained = True

    @precondition(lambda self: self.stopped)
    @rule()
    def worker_is_gone(self) -> None:
        """Nothing is left to do after the worker stopped; the run ends at teardown."""

    # ---- checks --------------------------------------------------------

    @invariant()
    def no_protocol_violations(self) -> None:
        assert_no_violations(self.agent, ignore=KNOWN_VIOLATIONS)

    @invariant()
    def scheduler_is_running_until_stopped(self) -> None:
        if not self.stopped:
            assert self.agent.running, f"WorkerScheduler exited with {self.agent.outcome!r}"

    def teardown(self) -> None:
        try:
            if not self.stopped:
                assert self.agent.running, f"WorkerScheduler exited with {self.agent.outcome!r}"
                for session_id in self._open_sessions():
                    self.service.close_session(session_id)
                ended = self.agent.wait_until(
                    lambda service: all(
                        service.sessions[session_id].ended for session_id in self.session_ids
                    ),
                    timeout=120,
                )
                assert ended, f"sessions did not end: {self.service.snapshot()['sessions']}"
                assert_service_drain_stops(self.agent, immediate=False)
            for session_id in self.session_ids:
                self.service.check_session_complete(session_id, worker_drained=self.worker_drained)
            assert_no_violations(self.agent, ignore=KNOWN_VIOLATIONS)
        finally:
            self._stop_agent()
            self.monkeypatch.undo()
            shutil.rmtree(self.tmp_path, ignore_errors=True)


WorkerProtocolMachine.TestCase.settings = settings(
    max_examples=int(os.environ.get("WORKER_PROTOCOL_EXAMPLES", "10")),
    stateful_step_count=int(os.environ.get("WORKER_PROTOCOL_STEPS", "10")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

test_worker_protocol_state_machine = WorkerProtocolMachine.TestCase
