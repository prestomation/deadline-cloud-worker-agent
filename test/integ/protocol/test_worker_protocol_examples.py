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
from deadline_worker_agent.aws.deadline import (
    DeadlineRequestWorkerNotFound,
    DeadlineRequestWorkerOfflineError,
)

from .harness import (
    JOB_ID,
    SECOND_JOB_ID,
    SECOND_QUEUE_ID,
    STEP_ID,
    AgentUnderTest,
    assert_no_violations,
    credential_failures,
    environment_id,
    leftover_processes,
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
    assert credential_failures(agent) == []
    assert leftover_processes() == []


def _open(
    agent: AgentUnderTest,
    *actions: tuple[str, dict],
    queue_id: str | None = None,
    job_id: str = JOB_ID,
) -> str:
    return agent.service.open_session(
        worker_id=agent.worker_id,
        queue_id=queue_id or agent.queue_id,
        job_id=job_id,
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


@pytest.mark.parametrize(
    "code",
    [
        # The name the current service sends in x-amzn-ErrorType for a 500.
        pytest.param("InternalServerException", id="current-service-name"),
        # The name in the service model, which the Rust data plane sends for some faults.
        pytest.param(
            "InternalServerErrorException",
            id="model-name",
            marks=pytest.mark.xfail(
                reason=(
                    "The service model names the HTTP 500 error InternalServerErrorException, "
                    "but the agent retries only a code named InternalServerException, so it "
                    "exits instead of retrying as the protocol says."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_internal_server_error_is_retried(agent: AgentUnderTest, code: str) -> None:
    agent.wait_for_syncs(1)

    # botocore retries a 5xx once on its own, so inject enough errors to reach the agent.
    agent.backend.inject_error("UpdateWorkerSchedule", code, count=4)
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
    # Both envExit actions are queued up front, before the first envEnter fails. The
    # current services do not do this (they queue envExit only after every envEnter is
    # reported), so this checks the agent's own rule, not a path seen in production.
    session_id = _open(
        agent,
        env_enter("BadEnter"),
        env_enter("Good"),
        ("envExit", {"environmentId": environment_id("Good")}),
        ("envExit", {"environmentId": environment_id("BadEnter")}),
    )
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


def test_sessions_on_two_queues_get_their_own_credentials(agent: AgentUnderTest) -> None:
    first = _open(agent, env_enter("Good"), task_run("0", 0))
    agent.service.close_session(first)
    _wait_ended(agent, first)
    second = _open(
        agent,
        env_enter("Good", SECOND_JOB_ID),
        task_run("0", 0),
        queue_id=SECOND_QUEUE_ID,
        job_id=SECOND_JOB_ID,
    )
    agent.service.close_session(second)
    _wait_ended(agent, second)

    queues = [
        queue_id for _, queue_id, outcome in agent.service.queue_role_calls if outcome == "OK"
    ]
    assert queues[:1] == [agent.queue_id] and SECOND_QUEUE_ID in queues
    assert credential_failures(agent) == []
    assert all(status == "SUCCEEDED" for _, status in _statuses(agent, second))
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


@pytest.mark.parametrize("code", ["InternalServerException", "ValidationException"])
def test_step_details_error_fails_the_task(agent: AgentUnderTest, code: str) -> None:
    agent.service.inject_entity_error(
        kind="stepDetails", job_id=JOB_ID, entity_id=STEP_ID, code=code, count=10
    )
    session_id = _open(agent, env_enter("Good"), task_run("0", 0), task_run("0", 0))

    _wait_ended_after_service_closes(agent, session_id)

    assert _statuses(agent, session_id) == [
        ("envEnter", "SUCCEEDED"),
        ("taskRun", "FAILED"),
        ("taskRun", "NEVER_ATTEMPTED"),
        ("envExit", "SUCCEEDED"),
    ]
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_environment_details_error_fails_the_env_enter(agent: AgentUnderTest) -> None:
    agent.service.inject_entity_error(
        kind="environmentDetails",
        job_id=JOB_ID,
        entity_id=environment_id("Good"),
        code="ResourceNotFoundException",
        count=10,
    )
    session_id = _open(agent, env_enter("Good"), task_run("0", 0))

    _wait_ended_after_service_closes(agent, session_id)

    assert _statuses(agent, session_id)[:2] == [
        ("envEnter", "FAILED"),
        ("taskRun", "NEVER_ATTEMPTED"),
    ]
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_job_details_error_fails_the_session(agent: AgentUnderTest) -> None:
    agent.service.inject_entity_error(
        kind="jobDetails",
        job_id=JOB_ID,
        entity_id=JOB_ID,
        code="InternalServerException",
        count=10,
    )
    session_id = _open(agent, env_enter("Good"), task_run("0", 0))

    _wait_ended_after_service_closes(agent, session_id)

    assert [status for _, status in _statuses(agent, session_id)][:2] != ["SUCCEEDED"] * 2
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_queue_role_not_found_fails_the_session(agent: AgentUnderTest) -> None:
    agent.service.inject_queue_role_error(agent.queue_id, "ResourceNotFoundException")
    session_id = _open(agent, env_enter("Good"), task_run("0", 0))

    _wait_ended_after_service_closes(agent, session_id)

    assert ("envEnter", "SUCCEEDED") not in _statuses(agent, session_id)
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_queue_role_access_denied_runs_without_credentials(agent: AgentUnderTest) -> None:
    agent.service.inject_queue_role_error(agent.queue_id, "AccessDeniedException")
    session_id = _open(agent, task_run("0", 0))
    agent.service.close_session(session_id)

    _wait_ended(agent, session_id)

    assert _statuses(agent, session_id) == [("taskRun", "SUCCEEDED")]
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


def test_not_responding_worker_stops_the_scheduler(agent: AgentUnderTest) -> None:
    session_id = _open(agent, task_run("30", 0))
    _wait_status(agent, agent.service.sessions[session_id].actions[0].session_action_id, "RUNNING")

    agent.service.mark_not_responding(agent.worker_id)

    assert isinstance(agent.join(timeout=60), DeadlineRequestWorkerOfflineError)
    assert leftover_processes() == []


def test_deleted_worker_stops_the_scheduler(agent: AgentUnderTest) -> None:
    session_id = _open(agent, task_run("30", 0))
    _wait_status(agent, agent.service.sessions[session_id].actions[0].session_action_id, "RUNNING")

    agent.service.forget_worker(agent.worker_id)

    assert isinstance(agent.join(timeout=60), DeadlineRequestWorkerNotFound)
    assert leftover_processes() == []


def test_worker_drain_during_env_enter(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("SlowEnter"), task_run("0", 0))
    enter_id = agent.service.sessions[session_id].actions[0].session_action_id
    _wait_status(agent, enter_id, "RUNNING")

    agent.scheduler.shutdown(grace_time=timedelta(seconds=5), fail_message="host is shutting down")
    assert agent.join(timeout=60) is None
    agent.set_status(WorkerStatus.STOPPED)

    assert _statuses(agent, session_id)[:2] == [
        ("envEnter", "INTERRUPTED"),
        ("taskRun", "NEVER_ATTEMPTED"),
    ]
    agent.service.check_session_complete(session_id, worker_drained=True)
    assert_no_violations(agent)


def test_canceling_one_env_exit_still_runs_the_next(agent: AgentUnderTest) -> None:
    session_id = _open(agent, env_enter("Good"), env_enter("SlowExit"))
    agent.service.close_session(session_id)
    # The exits are queued once both envEnter actions are reported.
    assert agent.wait_until(
        lambda service: len(service.sessions[session_id].actions) == 4, timeout=30
    )
    exit_slow, exit_good = (
        action.session_action_id for action in agent.service.sessions[session_id].actions[2:]
    )
    _wait_status(agent, exit_slow, "RUNNING")

    agent.service.cancel_action(exit_slow)
    _wait_ended(agent, session_id)

    assert agent.service.action(exit_slow).status == "CANCELED"
    assert agent.service.action(exit_good).status == "SUCCEEDED"
    assert_service_drain_stops(agent, immediate=False)
    assert_no_violations(agent)


_STOPPING_TIMER_BUG = pytest.mark.xfail(
    reason=(
        "_transition_to_stopping() creates a Timer to bound the STOPPING request but never "
        "starts it, so while UpdateWorker is throttled the drain keeps retrying and never "
        "reports the interrupted work."
    ),
    strict=True,
)


@pytest.mark.parametrize(
    ("code", "fields"),
    [
        pytest.param("ThrottlingException", {}, id="throttling", marks=_STOPPING_TIMER_BUG),
        pytest.param("ConflictException", {"reason": "STATUS_CONFLICT"}, id="status-conflict"),
        pytest.param("ResourceNotFoundException", {}, id="not-found"),
        pytest.param("AccessDeniedException", {}, id="access-denied"),
    ],
)
def test_worker_drain_finishes_when_update_worker_fails(
    agent: AgentUnderTest, code: str, fields: dict
) -> None:
    session_id = _open(agent, env_enter("Good"), task_run("30", 0))
    _wait_status(agent, agent.service.sessions[session_id].actions[1].session_action_id, "RUNNING")
    agent.backend.inject_error(
        "UpdateWorker", code, count=50, resourceId=agent.worker_id, resourceType="worker", **fields
    )

    agent.scheduler.shutdown(grace_time=timedelta(seconds=5), fail_message="host is shutting down")
    # The drain has a 5 second grace time, so it must finish well inside 20 seconds.
    assert agent.join(timeout=20) is None

    assert _statuses(agent, session_id)[1] == ("taskRun", "INTERRUPTED")
    assert_no_violations(agent)


@pytest.mark.xfail(
    reason=(
        "Session.stop() only sets an event. _drain_scheduler() waits on the stop() calls, "
        "not on the session threads, then sends its last UpdateWorkerSchedule. A session "
        "that is still starting reports NEVER_ATTEMPTED after that call, so the report "
        "is never sent."
    ),
    strict=True,
)
def test_worker_drain_reports_actions_of_a_session_that_is_still_starting(
    agent: AgentUnderTest,
) -> None:
    # A slow BatchGetJobEntity keeps the new session in its startup when the drain begins.
    agent.backend.operation_delays["BatchGetJobEntity"] = 2.0
    session_id = _open(agent, task_run("0", 0))
    assert agent.wait_until(
        lambda service: service.sessions[session_id].actions[0].first_sent_in_call is not None,
        timeout=10,
    )
    agent.wait_for_syncs(1)

    agent.scheduler.shutdown(grace_time=timedelta(seconds=5), fail_message="host is shutting down")
    assert agent.join(timeout=60) is None
    agent.set_status(WorkerStatus.STOPPED)

    assert _statuses(agent, session_id) == [("taskRun", "NEVER_ATTEMPTED")]
    assert_no_violations(agent)


@pytest.mark.xfail(
    reason=(
        "When a new session fails during setup (here, a jobDetails error), the agent "
        "reports its actions and forgets the session. If the next response still holds "
        "the session with a task that the service queued before it saw the failure, the "
        "agent sets the session up again and runs that task."
    ),
    strict=True,
)
def test_task_queued_before_a_setup_failure_is_not_run(agent: AgentUnderTest) -> None:
    # A slow BatchGetJobEntity holds the agent in session setup while the service
    # queues one more task.
    agent.backend.operation_delays["BatchGetJobEntity"] = 1.0
    agent.service.inject_entity_error(
        kind="jobDetails", job_id=JOB_ID, entity_id=JOB_ID, code="InternalServerException"
    )
    session_id = _open(agent, task_run("0", 0))
    first = agent.service.sessions[session_id].actions[0].session_action_id
    assert agent.wait_until(
        lambda service: service.action(first).first_sent_in_call is not None, timeout=10
    )
    (second,) = agent.service.add_actions(session_id, [task_run("0", 0)])
    assert agent.wait_until(lambda service: service.action(first).is_terminal, timeout=30)
    agent.backend.operation_delays.clear()

    _wait_status(agent, second, "NEVER_ATTEMPTED")
    assert_no_violations(agent)


@pytest.mark.xfail(
    reason=(
        "A task that reaches a session after an earlier task failed is reported FAILED "
        "('Session must be in the READY state to run a task.') instead of NEVER_ATTEMPTED."
    ),
    strict=True,
)
def test_task_queued_after_a_failure_is_reported_never_attempted(agent: AgentUnderTest) -> None:
    session_id = _open(agent, task_run("2", 1))
    first = agent.service.sessions[session_id].actions[0].session_action_id
    _wait_status(agent, first, "RUNNING")

    # Queued while the first task runs, so the agent receives it before the failure.
    (second,) = agent.service.add_actions(session_id, [task_run("0", 0)])
    assert agent.wait_until(lambda service: service.action(second).is_terminal, timeout=30)

    assert agent.service.action(second).status == "NEVER_ATTEMPTED"
    assert_no_violations(agent)
