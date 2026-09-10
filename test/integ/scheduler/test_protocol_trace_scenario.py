# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Runs the REAL WorkerScheduler + Session + openjd runtime through a scripted
multi-action session against a fake service client, with the protocol trace
enabled, and validates the trace the agent emits.

This is the trace-generation scenario for external protocol checkers: the
only fakes are at the service wire (UpdateWorkerSchedule / UpdateWorker /
BatchGetJobEntity responses) and the log destinations. Everything that emits
protocol trace records — the scheduler loop, the session action queue, the
session thread, and the action subprocesses — is the real implementation.

Scenario (one worker run, one session):
  1. The service assigns a session with ENV_ENTER + two TASK_RUNs.
  2. envEnter runs (echo) and succeeds; task-1 (sleep) starts.
  3. After the envEnter success report arrives, the service cancels the
     QUEUED task-2 -> settled NEVER_ATTEMPTED without timestamps.
  4. After the task-2 report arrives, the service cancels the RUNNING
     task-1 -> settled CANCELED.
  5. The service then assigns the ENV_EXIT, which runs and succeeds.
  6. The service removes the session; the test drains the worker.

The resulting trace file is also copied to the path in
DEADLINE_WORKER_PROTOCOL_TRACE_SCENARIO_OUT (when set) so external tooling
can replay it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

import deadline_worker_agent.worker_protocol_trace as trace_mod
from deadline_worker_agent.config import JobsRunAsUserOverride
from deadline_worker_agent.scheduler.scheduler import WorkerScheduler
from deadline_worker_agent.worker import Worker
from deadline_worker_agent.worker_protocol_trace import WorkerProtocolTrace, set_trace

if sys.platform == "win32":
    pytest.skip("POSIX-only scenario (uses /bin/sleep + echo)", allow_module_level=True)


FARM_ID = "farm-scenario"
FLEET_ID = "fleet-scenario"
WORKER_ID = "worker-scenario1111111111111111"
QUEUE_ID = "queue-scenario111111111111111111"
JOB_ID = "job-scenario1111111111111111111"
SESSION_ID = "session-scenario1111111111111111"
ENV_ID = "env-scenario:env1"
STEP_ID = "step-scenario1111111111111111111"
ACTION_ENV_ENTER = "sessionaction-scenario-enventer"
ACTION_TASK_1 = "sessionaction-scenario-task1"
ACTION_TASK_2 = "sessionaction-scenario-task2"
ACTION_ENV_EXIT = "sessionaction-scenario-envexit"

LOG_CONFIGURATION = {
    "logDriver": "awslogs",
    "options": {
        "logGroupName": "/scenario/worker",
        "logStreamName": "session-scenario",
    },
    "parameters": {"interval": "15"},
}


def _session_actions_initial() -> list[dict[str, Any]]:
    return [
        {
            "sessionActionId": ACTION_ENV_ENTER,
            "actionType": "ENV_ENTER",
            "environmentId": ENV_ID,
        },
        {
            "sessionActionId": ACTION_TASK_1,
            "actionType": "TASK_RUN",
            "stepId": STEP_ID,
            "taskId": "task-scenario-1",
            "parameters": {},
        },
        {
            "sessionActionId": ACTION_TASK_2,
            "actionType": "TASK_RUN",
            "stepId": STEP_ID,
            "taskId": "task-scenario-2",
            "parameters": {},
        },
    ]


def _session_action_env_exit() -> dict[str, Any]:
    return {
        "sessionActionId": ACTION_ENV_EXIT,
        "actionType": "ENV_EXIT",
        "environmentId": ENV_ID,
    }


class FakeService:
    """Scripted UpdateWorkerSchedule responses.

    Response semantics mirror the service: ``sessionActions`` is the
    session's CURRENT queue content (the worker's queue.replace() consumes
    it as a replacement list), containing the actions the service has not
    yet received any update for, minus actions being canceled. Cancels
    repeat until the terminal report arrives.
    """

    def __init__(self) -> None:
        self.reports: dict[str, str] = {}
        self.updates_seen: set[str] = set()
        self.delivered_env_exit = False
        self.session_removed = threading.Event()
        self.lock = threading.Lock()

    def update_worker_schedule(self, *, updated_session_actions=None, **kwargs) -> dict[str, Any]:
        with self.lock:
            for action_id, info in (updated_session_actions or {}).items():
                self.updates_seen.add(action_id)
                if completed := info.get("completedStatus"):
                    self.reports[action_id] = completed

            cancel_ids: list[str] = []
            pipeline: list[dict[str, Any]] = _session_actions_initial()
            include_session = True

            if ACTION_ENV_EXIT in self.reports:
                # Phase 3: everything settled; remove the session.
                include_session = False
                self.session_removed.set()
            elif self.delivered_env_exit:
                # Phase 2b: waiting for the envExit report.
                pipeline.append(_session_action_env_exit())
            elif ACTION_TASK_1 in self.reports and ACTION_TASK_2 in self.reports:
                # Phase 2: both cancels settled (CANCELED + NEVER_ATTEMPTED);
                # deliver the environment exit.
                self.delivered_env_exit = True
                pipeline.append(_session_action_env_exit())
            elif ACTION_ENV_ENTER in self.reports:
                # Phase 1: envEnter succeeded (task-1 is running/sleeping and
                # task-2 is queued); cancel BOTH tasks. The agent cancels the
                # RUNNING action immediately and settles the queued one
                # NEVER_ATTEMPTED when the canceled action ends (the service
                # does not accept out-of-order completion of queued cancels).
                cancel_ids = [ACTION_TASK_1, ACTION_TASK_2]

            assigned: dict[str, Any] = {}
            if include_session:
                # The session's current pipeline: everything without a
                # terminal report. Canceled-but-unsettled actions stay in the
                # list — the worker's queue must still hold them so its
                # cancel/cascade logic can settle them NEVER_ATTEMPTED. The
                # worker filters out its running action itself.
                session_actions = [
                    action for action in pipeline if action["sessionActionId"] not in self.reports
                ]
                assigned[SESSION_ID] = {
                    "queueId": QUEUE_ID,
                    "jobId": JOB_ID,
                    "sessionActions": session_actions,
                    "logConfiguration": LOG_CONFIGURATION,
                }
            return {
                "assignedSessions": assigned,
                "cancelSessionActions": {SESSION_ID: cancel_ids} if cancel_ids else {},
                "updateIntervalSeconds": 1,
            }


def _entity_response(identifier: dict[str, Any]) -> dict[str, Any]:
    """Serves BatchGetJobEntity requests for the scenario's job."""
    if "jobDetails" in identifier:
        return {
            "jobDetails": {
                "jobId": JOB_ID,
                "schemaVersion": "jobtemplate-2023-09",
                "logGroupName": "/scenario/worker",
            }
        }
    if "environmentDetails" in identifier:
        return {
            "environmentDetails": {
                "jobId": JOB_ID,
                "environmentId": identifier["environmentDetails"]["environmentId"],
                "schemaVersion": "jobtemplate-2023-09",
                "template": {
                    "name": "ScenarioEnv",
                    "script": {
                        "actions": {
                            "onEnter": {"command": "echo", "args": ["entering"]},
                            "onExit": {"command": "echo", "args": ["exiting"]},
                        }
                    },
                },
            }
        }
    if "stepDetails" in identifier:
        return {
            "stepDetails": {
                "jobId": JOB_ID,
                "stepId": identifier["stepDetails"]["stepId"],
                "schemaVersion": "jobtemplate-2023-09",
                "template": {
                    "name": "ScenarioStep",
                    "script": {
                        "actions": {
                            # Long enough that task-1 is still running when its
                            # cancel arrives; the cancel terminates it.
                            "onRun": {"command": "/bin/sleep", "args": ["120"]},
                        }
                    },
                },
                "dependencies": [],
            }
        }
    raise AssertionError(f"Unexpected entity identifier: {identifier}")


@pytest.fixture
def trace_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Installs a strict file-backed protocol trace for the scenario."""
    path = tmp_path / "worker-protocol-trace.jsonl"
    previous = set_trace(WorkerProtocolTrace(enabled=True, file_path=str(path), strict=True))
    yield path
    trace_mod.shutdown_trace()
    set_trace(previous)


def _read_trace(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # Every record the agent emits must be a canonical v1 envelope.
    for record in records:
        assert record["v"] == 1
        assert record["src"] == "worker-agent"
        assert isinstance(record["run"], str) and record["run"]
        assert isinstance(record["seq"], int) and record["seq"] >= 1
        assert record["corr"] is None or isinstance(record["corr"], str)
        assert isinstance(record["payload"], dict)
    return records


class TestProtocolTraceScenario:
    def test_multi_action_session_with_cancels_produces_protocol_trace(
        self, trace_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GIVEN a real scheduler wired to a scripted fake service
        monkeypatch.setenv("DEADLINE_CLOUD_TELEMETRY_OPT_OUT", "true")
        service = FakeService()

        deadline_client = MagicMock()

        # JobEntities introspects the boto service model for the max batch size.
        identifiers_field = MagicMock()
        identifiers_field.metadata = {"max": 5}
        operation_model = MagicMock()
        operation_model.input_shape.members = {"identifiers": identifiers_field}
        deadline_client._real_client._service_model.operation_model.return_value = operation_model

        def fake_batch_get_job_entity(*, identifiers, **kwargs) -> dict[str, Any]:
            return {
                "entities": [_entity_response(identifier) for identifier in identifiers],
                "errors": [],
            }

        deadline_client.batch_get_job_entity.side_effect = fake_batch_get_job_entity

        boto_session = MagicMock()
        boto_session.region_name = "us-west-2"
        boto_session.client.return_value.put_log_events.return_value = {
            "nextSequenceToken": "token-1",
        }

        session_root_dir = tmp_path / "sessions"
        session_root_dir.mkdir(parents=True, exist_ok=True)
        persistence_dir = tmp_path / "persistence"
        persistence_dir.mkdir(parents=True, exist_ok=True)

        scheduler = WorkerScheduler(
            deadline=deadline_client,
            farm_id=FARM_ID,
            fleet_id=FLEET_ID,
            worker_id=WORKER_ID,
            job_run_as_user_override=JobsRunAsUserOverride(run_as_agent=True),
            boto_session=boto_session,
            cleanup_session_user_processes=False,
            worker_persistence_dir=persistence_dir,
            worker_logs_dir=None,
            retain_session_dir=False,
            session_root_dir=session_root_dir,
        )

        worker = MagicMock(spec=Worker)
        worker._worker_id = WORKER_ID
        worker._run = scheduler.run

        def stop_when_session_removed() -> None:
            # Drain the worker once the service has taken the session back.
            assert service.session_removed.wait(timeout=120), "session never removed"
            scheduler.shutdown(fail_message="scenario complete")

        stopper = threading.Thread(target=stop_when_session_removed, daemon=True)

        # WHEN the real worker loop runs the scenario end to end
        with (
            patch(
                "deadline_worker_agent.scheduler.scheduler.update_worker_schedule",
                side_effect=lambda **kw: service.update_worker_schedule(
                    updated_session_actions=kw.get("updated_session_actions")
                ),
            ),
            patch(
                "deadline_worker_agent.scheduler.scheduler.update_worker",
                return_value={},
            ),
        ):
            stopper.start()
            Worker.run(worker)  # the real hook: emits worker.start/worker.stop
            stopper.join(timeout=10)

        # THEN the agent produced a complete protocol history
        records = _read_trace(trace_file)
        events = [(r["event"], r["corr"]) for r in records]

        def index_of(event: str, corr: str | None) -> int:
            assert (event, corr) in events, f"missing {event} {corr}: {events}"
            return events.index((event, corr))

        # One worker run boot record with the worker id.
        assert events[0] == ("worker.start", None)
        assert records[0]["payload"] == {"worker_id": WORKER_ID}
        assert events[-1] == ("worker.stop", None)
        run_ids = {r["run"] for r in records}
        assert len(run_ids) == 1
        seqs = [r["seq"] for r in records]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

        # Every action was announced as newly assigned exactly once.
        for action_id in (ACTION_ENV_ENTER, ACTION_TASK_1, ACTION_TASK_2, ACTION_ENV_EXIT):
            assigned = [e for e in events if e == ("action.assigned", action_id)]
            assert len(assigned) == 1, f"{action_id}: {assigned}"

        # The session started after its actions were observed.
        assert index_of("session.start", SESSION_ID) > index_of("action.assigned", ACTION_ENV_ENTER)

        # envEnter ran and succeeded before task-1 started (sequential pipeline).
        assert (
            index_of("action.start", ACTION_ENV_ENTER)
            < index_of("action.complete", ACTION_ENV_ENTER)
            < index_of("action.start", ACTION_TASK_1)
        )

        # task-2's cancel was delivered and it NEVER started.
        assert ("action.start", ACTION_TASK_2) not in events
        assert index_of("action.cancel", ACTION_TASK_2) < index_of("action.complete", ACTION_TASK_2)
        task2_settle = records[index_of("action.complete", ACTION_TASK_2)]
        assert task2_settle["payload"]["status"] == "NEVER_ATTEMPTED"
        assert task2_settle["payload"]["session_id"] == SESSION_ID

        # task-1 was canceled and settled CANCELED after it started. (The
        # service repeats cancels until the terminal report is acked, and the
        # first delivery can race task-1's start, so only start-before-settle
        # is asserted.)
        assert ("action.cancel", ACTION_TASK_1) in events
        task1_settle_idx = index_of("action.complete", ACTION_TASK_1)
        assert index_of("action.start", ACTION_TASK_1) < task1_settle_idx
        task1_settle = records[task1_settle_idx]
        assert task1_settle["payload"]["status"] == "CANCELED"

        # envExit still ran after the failure cascade, and succeeded.
        env_exit_start = index_of("action.start", ACTION_ENV_EXIT)
        assert env_exit_start > index_of("action.complete", ACTION_TASK_1)
        env_exit_settle = records[index_of("action.complete", ACTION_ENV_EXIT)]
        assert env_exit_settle["payload"]["status"] == "SUCCEEDED"
        assert records[env_exit_start]["payload"]["env_id"] == ENV_ID

        # Status reports were sent for every terminal decision, and the
        # NEVER_ATTEMPTED report carried no timestamps.
        report_statuses = {
            r["corr"]: r["payload"]
            for r in records
            if r["event"] == "action.report" and r["payload"].get("status")
        }
        assert report_statuses[ACTION_ENV_ENTER]["status"] == "SUCCEEDED"
        assert report_statuses[ACTION_TASK_1]["status"] == "CANCELED"
        assert report_statuses[ACTION_TASK_2]["status"] == "NEVER_ATTEMPTED"
        assert report_statuses[ACTION_TASK_2]["has_timestamps"] is False
        assert report_statuses[ACTION_ENV_EXIT]["status"] == "SUCCEEDED"
        for payload in report_statuses.values():
            assert payload["session_id"] == SESSION_ID

        # The session completed and the worker drained.
        assert ("session.complete", SESSION_ID) in events
        assert ("drain.requested", None) in events
        assert index_of("drain.complete", None) > index_of("drain.start", None)

        # Export the capture for external tooling (PObserve replay).
        if out := os.environ.get("DEADLINE_WORKER_PROTOCOL_TRACE_SCENARIO_OUT"):
            out_path = Path(out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(trace_file, out_path)
