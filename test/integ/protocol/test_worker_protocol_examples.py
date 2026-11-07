# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Fixed scenarios for the worker protocol, run against the hermetic mock service.

Each test drives one path in ``docs/dev/worker_api_protocol.md``. The mock service
checks every report the agent sends against the protocol rules, so each test ends by
asserting that the service saw no violations. The property-based test in
``test_worker_protocol_state_machine.py`` explores many more orderings.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Generator

import pytest

from deadline_worker_agent.api_models import WorkerStatus
from deadline_worker_agent.aws.deadline import DeadlineRequestWorkerOfflineError

from .harness import (
    JOB_ID,
    AgentUnderTest,
    assert_no_violations,
    assert_service_drain_stops,
    env_enter,
    start_agent,
    task_run,
)

pytestmark = pytest.mark.timeout(180)


@pytest.fixture
def agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[AgentUnderTest, None, None]:
    agent, stop = start_agent(tmp_path, monkeypatch)
    try:
        yield agent
    finally:
        stop()


def _open(agent: AgentUnderTest, *actions: tuple[str, dict]) -> str:
    return agent.service.open_session(
        worker_id=agent.worker_id,
        queue_id=agent.queue_id,
        job_id=JOB_ID,
        actions=actions,
    )


def _statuses(agent: AgentUnderTest, session_id: str) -> list[tuple[str, str]]:
    return [(action.kind, action.status) for action in agent.service.sessions[session_id].actions]


def _wait_ended(agent: AgentUnderTest, session_id: str, timeout: float = 60) -> None:
    assert agent.wait_until(lambda service: service.sessions[session_id].ended, timeout), (
        f"session did not end: {_statuses(agent, session_id)}"
    )


def _wait_status(agent: AgentUnderTest, action_id: str, status: str, timeout: float = 30) -> None:
    assert agent.wait_until(lambda service: service.action(action_id).status == status, timeout), (
        f"{action_id} is {agent.service.action(action_id).status}, not {status}"
    )


def test_session_runs_in_order_and_exits_its_environment(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), task_run("0", 0), task_run("0", 0))
    agent.service.close_session(session_id)

    _wait_ended(agent, session_id)

    assert _statuses(agent, session_id) == [
        ("envEnter", "SUCCEEDED"),
        ("taskRun", "SUCCEEDED"),
        ("taskRun", "SUCCEEDED"),
        ("envExit", "SUCCEEDED"),
    ]
    agent.service.check_session_complete(session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_failed_task_skips_later_tasks_but_still_exits_environment(
    agent: AgentUnderTest,
) -> None:
    session_id = _open(
        agent, env_enter("Good"), task_run("0", 1), task_run("0", 0), task_run("0", 0)
    )

    _wait_ended_after_service_closes(agent, session_id)

    assert _statuses(agent, session_id) == [
        ("envEnter", "SUCCEEDED"),
        ("taskRun", "FAILED"),
        ("taskRun", "NEVER_ATTEMPTED"),
        ("taskRun", "NEVER_ATTEMPTED"),
        ("envExit", "SUCCEEDED"),
    ]
    agent.service.check_session_complete(session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def _wait_ended_after_service_closes(agent: AgentUnderTest, session_id: str) -> None:
    # After a failure the mock service queues the envExit actions itself; the test
    # closes the session once nothing else is left to run.
    assert agent.wait_until(
        lambda service: (
            service.sessions[session_id].blocked_at is not None
            or service.sessions[session_id].all_terminal
        ),
        timeout=60,
    )
    agent.service.close_session(session_id)
    _wait_ended(agent, session_id)


def test_failed_env_enter_still_runs_its_env_exit(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), env_enter("BadEnter"), task_run("0", 0))

    _wait_ended_after_service_closes(agent, session_id)

    assert _statuses(agent, session_id) == [
        ("envEnter", "SUCCEEDED"),
        ("envEnter", "FAILED"),
        ("taskRun", "NEVER_ATTEMPTED"),
        ("envExit", "SUCCEEDED"),
        ("envExit", "SUCCEEDED"),
    ]
    agent.service.check_session_complete(session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_failed_env_exit_does_not_stop_the_next_env_exit(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), env_enter("BadExit"), task_run("0", 0))
    agent.service.close_session(session_id)

    _wait_ended(agent, session_id)

    assert _statuses(agent, session_id)[-2:] == [
        ("envExit", "FAILED"),
        ("envExit", "SUCCEEDED"),
    ]
    agent.service.check_session_complete(session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_service_cancel_of_running_task(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), task_run("30", 0), task_run("0", 0))
    running_task = agent.service.sessions[session_id].actions[1].session_action_id
    _wait_status(agent, running_task, "RUNNING")

    agent.service.cancel_action(running_task)
    agent.service.close_session(session_id)
    _wait_ended(agent, session_id)

    assert _statuses(agent, session_id) == [
        ("envEnter", "SUCCEEDED"),
        ("taskRun", "CANCELED"),
        ("taskRun", "NEVER_ATTEMPTED"),
        ("envExit", "SUCCEEDED"),
    ]
    agent.service.check_session_complete(session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_immediate_service_drain_cancels_work_and_stops(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), task_run("30", 0), task_run("30", 0))
    _wait_status(agent, agent.service.sessions[session_id].actions[1].session_action_id, "RUNNING")

    assert_service_drain_stops(agent, immediate=True)

    kinds_and_statuses = _statuses(agent, session_id)
    assert kinds_and_statuses[1:3] == [("taskRun", "CANCELED"), ("taskRun", "NEVER_ATTEMPTED")]
    assert kinds_and_statuses[-1] == ("envExit", "SUCCEEDED")
    agent.service.check_session_complete(session_id)
    assert_no_violations(agent)


def test_worker_initiated_drain_reports_everything_before_stopping(
    agent: AgentUnderTest,
) -> None:
    session_id = _open(agent, env_enter("Good"), task_run("30", 0), task_run("0", 0))
    running_task = agent.service.sessions[session_id].actions[1].session_action_id
    _wait_status(agent, running_task, "RUNNING")

    agent.scheduler.shutdown(grace_time=timedelta(seconds=5), fail_message="host is shutting down")
    assert agent.join(timeout=60) is None
    agent.set_status(WorkerStatus.STOPPED)

    worker = next(iter(agent.service.workers.values()))
    assert worker.status_history[-2:] == ["STOPPING", "STOPPED"]
    # The service may queue an envExit in its answer to the last call; the agent
    # does not run it in an expedited drain.
    assert _statuses(agent, session_id)[:3] == [
        ("envEnter", "SUCCEEDED"),
        ("taskRun", "INTERRUPTED"),
        ("taskRun", "NEVER_ATTEMPTED"),
    ]
    agent.service.check_session_complete(session_id, worker_drained=True)
    assert_no_violations(agent)


@pytest.mark.parametrize(
    ("code", "fields"),
    [
        pytest.param("ThrottlingException", {}, id="throttling"),
        pytest.param(
            "ConflictException",
            {"reason": "CONCURRENT_MODIFICATION", "resourceType": "worker"},
            id="concurrent-modification",
        ),
    ],
)
def test_retryable_schedule_errors_do_not_lose_reports(
    agent: AgentUnderTest, code: str, fields: dict
) -> None:
    session_id = _open(agent, task_run("0", 0), task_run("0.3", 0))
    first_task = agent.service.sessions[session_id].actions[0].session_action_id
    agent.wait_until(lambda service: service.action(first_task).status != "ASSIGNED", timeout=30)

    agent.backend.inject_error(
        "UpdateWorkerSchedule", code, count=2, resourceId=agent.worker_id, **fields
    )
    agent.service.close_session(session_id)
    _wait_ended(agent, session_id)

    assert _statuses(agent, session_id) == [("taskRun", "SUCCEEDED"), ("taskRun", "SUCCEEDED")]
    assert agent.backend.injected_errors["UpdateWorkerSchedule"] == []
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


@pytest.mark.xfail(
    reason=(
        "The service model names the HTTP 500 error InternalServerErrorException, but "
        "the agent retries only a code named InternalServerException, so it exits "
        "instead of retrying as the protocol says."
    ),
    strict=True,
)
def test_internal_server_error_is_retried(agent: AgentUnderTest) -> None:
    agent.wait_for_syncs(1)

    # botocore retries a 5xx once on its own, so inject enough errors to reach the agent.
    agent.backend.inject_error("UpdateWorkerSchedule", "InternalServerErrorException", count=4)
    agent.wait_for_syncs(2, timeout=30)

    assert agent.running, f"scheduler exited with {agent.outcome!r}"


def test_status_conflict_on_the_worker_ends_the_scheduler(agent: AgentUnderTest) -> None:
    agent.wait_for_syncs(1)

    agent.backend.inject_error(
        "UpdateWorkerSchedule",
        "ConflictException",
        reason="STATUS_CONFLICT",
        resourceId=agent.worker_id,
        resourceType="worker",
    )

    assert isinstance(agent.join(timeout=30), DeadlineRequestWorkerOfflineError)


@pytest.mark.xfail(
    reason=(
        "After an envEnter fails, the protocol says to report the envExit of an environment "
        "that was never entered as NEVER_ATTEMPTED. The agent keeps every envExit in the "
        "queue, tries to run it, and reports FAILED: 'not the inner-most active environment'."
    ),
    strict=True,
)
def test_env_exit_for_an_environment_never_entered_is_not_run(agent: AgentUnderTest) -> None:
    # The service closes the session before the agent starts, so both envExit actions
    # are already queued when the first envEnter fails.
    session_id = _open(agent, env_enter("BadEnter"), env_enter("Good"))
    agent.service.close_session(session_id)

    _wait_ended(agent, session_id)

    statuses = _statuses(agent, session_id)
    assert statuses[:2] == [("envEnter", "FAILED"), ("envEnter", "NEVER_ATTEMPTED")]
    exit_good = next(
        action
        for action in agent.service.sessions[session_id].actions
        if action.kind == "envExit" and action.environment_id.endswith(":Good")
    )
    assert exit_good.status == "NEVER_ATTEMPTED"
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)
