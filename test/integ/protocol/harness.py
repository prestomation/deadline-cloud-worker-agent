# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Run the real WorkerScheduler against the hermetic Deadline Cloud service in
``deadline_test_fixtures.deadline_mock``.

The service runs in this process on 127.0.0.1. The agent talks to it through a real
boto3 ``deadline`` client, so every request and response crosses the wire format and
is checked against the botocore service model. Session actions run real Open Job
Description sessions with short Python subprocesses.

Two things are replaced, because they are outside the worker protocol:

* Session logs go to a ``logging.NullHandler`` instead of CloudWatch Logs.
* Telemetry is opted out with ``DEADLINE_CLOUD_TELEMETRY_OPT_OUT``.
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import boto3
import pytest
from botocore.config import Config

from deadline_worker_agent.api_models import WorkerStatus
from deadline_worker_agent.aws.deadline import update_worker
from deadline_worker_agent.boto import DEADLINE_BOTOCORE_CONFIG, DeadlineClient
from deadline_worker_agent.config import JobsRunAsUserOverride
from deadline_worker_agent.errors import ServiceShutdown
from deadline_worker_agent.scheduler import WorkerScheduler
from deadline_worker_agent.sessions.log_config import LogConfiguration

deadline_mock = pytest.importorskip(
    "deadline_test_fixtures.deadline_mock",
    reason="needs deadline-cloud-test-fixtures with the mock worker APIs",
)
if not hasattr(deadline_mock, "MockWorkerScheduler"):
    pytest.skip(
        "needs deadline-cloud-test-fixtures with MockWorkerScheduler", allow_module_level=True
    )

FLEET_ID = "fleet-00000000000000000000000000000001"
JOB_ID = "job-00000000000000000000000000000001"
STEP_ID = "step-00000000000000000000000000000001"
SCHEMA = "jobtemplate-2023-09"

_TASK_SCRIPT = "import sys, time; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))"


def environment_id(name: str) -> str:
    return f"JOB:{JOB_ID}:{name}"


def _python_action(code: str, *args: str) -> dict[str, Any]:
    return {"command": sys.executable, "args": ["-c", code, *args]}


def _environment(name: str, enter_exit_code: int, exit_exit_code: int = 0) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA,
        "template": {
            "name": name,
            "script": {
                "actions": {
                    "onEnter": _python_action(f"raise SystemExit({enter_exit_code})"),
                    "onExit": _python_action(f"raise SystemExit({exit_exit_code})"),
                }
            },
        },
    }


ENVIRONMENTS: dict[str, dict[str, Any]] = {
    environment_id("Good"): _environment("Good", 0),
    environment_id("BadEnter"): _environment("BadEnter", 1),
    environment_id("BadExit"): _environment("BadExit", 0, exit_exit_code=1),
}
"""Environments a test can enter. The name says how the environment behaves."""

STEP = {
    "schemaVersion": SCHEMA,
    "template": {
        "name": "Work",
        "parameterSpace": {
            "taskParameterDefinitions": [
                {"name": "Sleep", "type": "STRING", "range": ["0", "0.3", "2", "30"]},
                {"name": "Exit", "type": "INT", "range": [0, 1]},
            ]
        },
        "script": {
            "actions": {
                "onRun": _python_action(_TASK_SCRIPT, "{{Task.Param.Sleep}}", "{{Task.Param.Exit}}")
            }
        },
    },
}


def task_run(sleep: str, exit_code: int) -> tuple[str, dict[str, Any]]:
    """A taskRun action that sleeps ``sleep`` seconds, then exits with ``exit_code``."""
    return (
        "taskRun",
        {
            "stepId": STEP_ID,
            "taskId": f"{STEP_ID}-{sleep.replace('.', '')}{exit_code}",
            "parameters": {"Sleep": {"string": sleep}, "Exit": {"int": str(exit_code)}},
        },
    )


def env_enter(name: str) -> tuple[str, dict[str, Any]]:
    return ("envEnter", {"environmentId": environment_id(name)})


@dataclass
class AgentUnderTest:
    """One WorkerScheduler running against one mock service."""

    backend: Any
    deadline: DeadlineClient
    worker_id: str
    scheduler: WorkerScheduler
    thread: threading.Thread
    outcome: list[BaseException | None] = field(default_factory=list)

    @property
    def farm_id(self) -> str:
        return str(self.backend.farm_id)

    @property
    def queue_id(self) -> str:
        return str(self.backend.queue_id)

    @property
    def service(self) -> Any:
        return self.backend.worker_scheduler

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def wait_for_syncs(self, count: int = 1, timeout: float = 20) -> bool:
        return self.service.wait_for_schedule_calls(self.worker_id, count, timeout)

    def wait_until(self, predicate: Callable[[Any], bool], timeout: float = 30) -> bool:
        return self.service.wait_until(predicate, timeout)

    def join(self, timeout: float = 60) -> Optional[BaseException]:
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise AssertionError(f"WorkerScheduler did not exit within {timeout}s")
        return self.outcome[0] if self.outcome else None

    def set_status(self, status: WorkerStatus) -> None:
        update_worker(
            deadline_client=self.deadline,
            farm_id=self.farm_id,
            fleet_id=FLEET_ID,
            worker_id=self.worker_id,
            status=status,
        )


def _null_remote_handler(self: LogConfiguration, *, boto_session: Any) -> logging.Handler:
    return logging.NullHandler()


def start_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, update_interval_seconds: int = 1
) -> tuple[AgentUnderTest, Callable[[], None]]:
    """Start the mock service and a WorkerScheduler. Returns the agent and a stop function."""
    monkeypatch.setenv("DEADLINE_CLOUD_TELEMETRY_OPT_OUT", "true")
    monkeypatch.setattr(LogConfiguration, "create_remote_handler", _null_remote_handler)
    for name in ("AWS_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_DEADLINE"):
        monkeypatch.delenv(name, raising=False)

    # A queue with no IAM role and no job attachments, so no session needs AWS access.
    queue = dict(deadline_mock.MockDeadlineScenario().queue)
    queue.pop("jobAttachmentSettings", None)
    queue.pop("roleArn", None)
    backend = deadline_mock.MockDeadlineBackend(deadline_mock.MockDeadlineScenario(queue=queue))
    backend.worker_scheduler.update_interval_seconds = update_interval_seconds
    backend.worker_scheduler.add_job(
        job_id=JOB_ID,
        job_details={"logGroupName": "/aws/deadline/mock", "schemaVersion": SCHEMA},
        steps={STEP_ID: STEP},
        environments=ENVIRONMENTS,
    )
    server, endpoint, server_thread = deadline_mock.start_server(backend)

    boto_session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-west-2",
    )
    client = boto_session.client(
        "deadline",
        endpoint_url=endpoint,
        # The agent's own client config, except that the worker APIs' "scheduling."
        # host prefix is not added to the loopback endpoint.
        config=DEADLINE_BOTOCORE_CONFIG.merge(Config(inject_host_prefix=False)),
    )
    deadline = DeadlineClient(client)
    worker_id = deadline.create_worker(farmId=backend.farm_id, fleetId=FLEET_ID)["workerId"]
    update_worker(
        deadline_client=deadline,
        farm_id=backend.farm_id,
        fleet_id=FLEET_ID,
        worker_id=worker_id,
        status=WorkerStatus.STARTED,
    )

    for directory in ("persistence", "logs", "sessions"):
        (tmp_path / directory).mkdir()
    scheduler = WorkerScheduler(
        farm_id=backend.farm_id,
        fleet_id=FLEET_ID,
        worker_id=worker_id,
        deadline=deadline,
        job_run_as_user_override=JobsRunAsUserOverride(run_as_agent=True),
        boto_session=boto_session,  # type: ignore[arg-type]
        cleanup_session_user_processes=False,
        worker_persistence_dir=tmp_path / "persistence",
        worker_logs_dir=tmp_path / "logs",
        session_root_dir=tmp_path / "sessions",
    )
    outcome: list[BaseException | None] = []

    def run() -> None:
        try:
            scheduler.run()
        except BaseException as error:  # noqa: BLE001 - the test inspects the outcome
            outcome.append(error)
        else:
            outcome.append(None)

    thread = threading.Thread(target=run, name="WorkerSchedulerUnderTest", daemon=True)
    agent = AgentUnderTest(
        backend=backend,
        deadline=deadline,
        worker_id=worker_id,
        scheduler=scheduler,
        thread=thread,
        outcome=outcome,
    )
    thread.start()

    def stop() -> None:
        if thread.is_alive():
            scheduler.shutdown(grace_time=timedelta(seconds=5))
            thread.join(timeout=30)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    return agent, stop


def assert_service_drain_stops(agent: AgentUnderTest, *, immediate: bool) -> None:
    """Drain from the service side and check the agent shuts down cleanly."""
    agent.service.drain(agent.worker_id, immediate=immediate)
    outcome = agent.join(timeout=60)
    assert isinstance(outcome, ServiceShutdown), f"scheduler ended with {outcome!r}"
    agent.set_status(WorkerStatus.STOPPED)


def assert_no_violations(agent: AgentUnderTest, *, ignore: frozenset[str] = frozenset()) -> None:
    """Fail if the service saw a protocol violation whose rule is not in ``ignore``."""
    violations = [
        str(violation) for violation in agent.service.violations if violation.rule not in ignore
    ]
    assert violations == [], "worker protocol violations:\n" + "\n".join(violations)
