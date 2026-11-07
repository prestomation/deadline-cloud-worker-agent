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
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import boto3
import psutil
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
"""The job on the scenario's queue."""
SECOND_QUEUE_ID = "queue-0000000000000000000000000000000e"
SECOND_JOB_ID = "job-00000000000000000000000000000002"
"""A job on a second queue, to test that queue credentials never mix."""
STEP_ID = "step-00000000000000000000000000000001"
SCHEMA = "jobtemplate-2023-09"

CREDENTIALS_MISMATCH_EXIT = 3
"""A script exits with this code when its AWS credentials belong to another queue."""
CREDENTIALS_BROKEN_EXIT = 4
"""A script exits with this code when its credential process is set but fails."""

# Every script first checks the AWS credentials that the agent gave it. The mock service
# names each queue's credentials ASIAQUEUE<last 8 characters of the queue ID>.
_CHECK_CREDENTIALS = f"""
import configparser, json, os, subprocess, sys, time
profile = os.environ.get("AWS_PROFILE")
if profile:
    config = configparser.ConfigParser()
    config.read(os.environ["AWS_CONFIG_FILE"])
    try:
        command = config["profile " + profile]["credential_process"]
        output = subprocess.run(command, shell=True, capture_output=True, check=True).stdout
        key = json.loads(output)["AccessKeyId"]
    except Exception as error:
        print("credential process failed:", error)
        sys.exit({CREDENTIALS_BROKEN_EXIT})
    expected = "ASIAQUEUE" + os.environ["DEADLINE_QUEUE_ID"][-8:].upper()
    if key != expected:
        print("credentials for the wrong queue:", key, "expected", expected)
        sys.exit({CREDENTIALS_MISMATCH_EXIT})
"""
_SLEEP_THEN_EXIT = _CHECK_CREDENTIALS + "time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))"

PROCESS_MARKER = f"worker-protocol-test-process-{os.getpid()}"
"""Passed to every script, so a test can find processes the agent left running."""


def environment_id(name: str, job_id: str = JOB_ID) -> str:
    return f"JOB:{job_id}:{name}"


def _python_action(sleep: str, exit_code: int | str) -> dict[str, Any]:
    return {
        "command": sys.executable,
        "args": ["-c", _SLEEP_THEN_EXIT, sleep, str(exit_code), PROCESS_MARKER],
    }


def _environment(
    name: str,
    *,
    enter_exit: int = 0,
    exit_exit: int = 0,
    enter_sleep: str = "0",
    exit_sleep: str = "0",
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA,
        "template": {
            "name": name,
            "script": {
                "actions": {
                    "onEnter": _python_action(enter_sleep, enter_exit),
                    "onExit": _python_action(exit_sleep, exit_exit),
                }
            },
        },
    }


ENVIRONMENT_NAMES = ("Good", "BadEnter", "BadExit", "SlowEnter", "SlowExit")
"""Environments a test can enter. The name says how the environment behaves."""


def _environments(job_id: str) -> dict[str, dict[str, Any]]:
    return {
        environment_id("Good", job_id): _environment("Good"),
        environment_id("BadEnter", job_id): _environment("BadEnter", enter_exit=1),
        environment_id("BadExit", job_id): _environment("BadExit", exit_exit=1),
        environment_id("SlowEnter", job_id): _environment("SlowEnter", enter_sleep="30"),
        environment_id("SlowExit", job_id): _environment("SlowExit", exit_sleep="30"),
    }


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
            "actions": {"onRun": _python_action("{{Task.Param.Sleep}}", "{{Task.Param.Exit}}")}
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


def env_enter(name: str, job_id: str = JOB_ID) -> tuple[str, dict[str, Any]]:
    return ("envEnter", {"environmentId": environment_id(name, job_id)})


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

    # Queues with IAM roles but no job attachments: sessions get queue credentials, but
    # nothing talks to S3.
    queue = dict(deadline_mock.MockDeadlineScenario().queue)
    queue.pop("jobAttachmentSettings", None)
    backend = deadline_mock.MockDeadlineBackend(deadline_mock.MockDeadlineScenario(queue=queue))
    service = backend.worker_scheduler
    service.update_interval_seconds = update_interval_seconds
    for queue_id, job_id in ((backend.queue_id, JOB_ID), (SECOND_QUEUE_ID, SECOND_JOB_ID)):
        role_arn = f"arn:aws:iam::123456789012:role/Queue{queue_id[-4:]}"
        service.queue_role_arns[queue_id] = role_arn
        service.add_job(
            job_id=job_id,
            job_details={
                "logGroupName": "/aws/deadline/mock",
                "schemaVersion": SCHEMA,
                "queueRoleArn": role_arn,
            },
            steps={STEP_ID: STEP},
            environments=_environments(job_id),
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
        # Clear injected errors first, so a stuck retry loop can finish.
        backend.injected_errors.clear()
        if thread.is_alive():
            scheduler.shutdown(grace_time=timedelta(seconds=5))
            thread.join(timeout=90)
        if thread.is_alive():
            # SessionMap is a process-wide singleton, so a scheduler that is still running
            # would change the sessions of the next test. Stop the run instead.
            pytest.exit("WorkerScheduler did not stop; later tests would share its state")
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    return agent, stop


def assert_service_drain_stops(agent: AgentUnderTest, *, immediate: bool) -> None:
    """Drain from the service side and check the agent shuts down cleanly."""
    agent.service.drain(agent.worker_id, immediate=immediate)
    # An eventual drain waits for running work: up to 30 seconds for SlowEnter and 30 for
    # SlowExit, plus retry backoff.
    outcome = agent.join(timeout=150)
    assert isinstance(outcome, ServiceShutdown), f"scheduler ended with {outcome!r}"
    agent.set_status(WorkerStatus.STOPPED)


def leftover_processes() -> list[str]:
    """Command lines of test scripts that are still running."""
    found = []
    for process in psutil.process_iter(["cmdline"]):
        cmdline = process.info.get("cmdline") or []
        if PROCESS_MARKER in cmdline:
            found.append(" ".join(cmdline[-3:]))
    return found


def credential_failures(agent: AgentUnderTest) -> list[str]:
    """Actions whose script saw another queue's credentials or broken credentials."""
    return [
        f"{session.session_id}/{action.session_action_id} {action.kind} exited "
        f"{action.process_exit_code}: {action.progress_message}"
        for session in agent.service.sessions.values()
        for action in session.actions
        if action.process_exit_code in (CREDENTIALS_MISMATCH_EXIT, CREDENTIALS_BROKEN_EXIT)
    ]


def assert_no_violations(agent: AgentUnderTest, *, ignore: frozenset[str] = frozenset()) -> None:
    """Fail if the service saw a protocol violation whose rule is not in ``ignore``."""
    violations = [
        str(violation) for violation in agent.service.violations if violation.rule not in ignore
    ]
    assert violations == [], "worker protocol violations:\n" + "\n".join(violations)
