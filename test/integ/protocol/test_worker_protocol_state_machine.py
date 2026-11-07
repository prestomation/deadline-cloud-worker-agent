# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Property-based test of the worker protocol.

Hypothesis plays the service. Each step is something the service can do: assign a
session on one of two queues, add tasks, cancel an action, close a session, return
errors (for a whole call, for one job entity, or for queue credentials), lose track of
the worker, drain the worker, or let time pass. The worker can also drain itself. The
real WorkerScheduler runs against the hermetic mock service in
``deadline_test_fixtures.deadline_mock``.

After every step:

* the mock service's protocol checker has no violations;
* no script saw AWS credentials of another queue, or a broken credential process;
* the scheduler is still running, unless the run stopped it on purpose.

At the end, every session is complete (each action reported, each entered environment
exited) and no script process is left running.

When a run fails, Hypothesis shrinks it to a short sequence of steps and prints it.

Environment variables:

* ``WORKER_PROTOCOL_EXAMPLES``: number of runs (default 10).
* ``WORKER_PROTOCOL_STEPS``: maximum steps in one run (default 10).
* ``WORKER_PROTOCOL_TIMEOUT``: pytest timeout in seconds (default 3600).

To split a large run over processes, give each one its own ``--hypothesis-seed``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from deadline_worker_agent.api_models import WorkerStatus
from deadline_worker_agent.aws.deadline import (
    DeadlineRequestWorkerNotFound,
    DeadlineRequestWorkerOfflineError,
)

from .harness import (
    ENVIRONMENT_NAMES,
    JOB_ID,
    SECOND_JOB_ID,
    SECOND_QUEUE_ID,
    STEP_ID,
    assert_no_violations,
    assert_service_drain_stops,
    credential_failures,
    env_enter,
    environment_id,
    leftover_processes,
    start_agent,
    task_run,
)

pytestmark = pytest.mark.timeout(int(os.environ.get("WORKER_PROTOCOL_TIMEOUT", "3600")))

KNOWN_VIOLATIONS = frozenset(
    {
        "exit-without-enter",
        "ran-after-setup-failure",
        "rejected-instead-of-never-attempted",
    }
)
"""Rules the agent is known to break. Each one has a strict xfail test in
test_worker_protocol_examples.py, so a fix shows up there. Ignoring them here lets the
state machine keep looking for new problems."""

KNOWN_VIOLATIONS_AFTER_WORKER_DRAIN = KNOWN_VIOLATIONS | {
    "stopped-with-open-actions",
    "action-never-completed",
}
"""A worker drain can send its last report before a starting session reports its
actions (test_worker_drain_reports_actions_of_a_session_that_is_still_starting)."""

_TASKS = st.builds(
    task_run,
    sleep=st.sampled_from(["0", "0.3", "2"]),
    exit_code=st.sampled_from([0, 0, 0, 1]),
)
_ENVIRONMENTS = st.lists(st.sampled_from(ENVIRONMENT_NAMES), unique=True, max_size=2)
_ERROR_COUNT = st.integers(min_value=1, max_value=2)
_RETRYABLE_ERRORS: st.SearchStrategy[tuple[str, str, dict[str, str], int]] = st.one_of(
    st.tuples(
        st.sampled_from(["UpdateWorkerSchedule", "BatchGetJobEntity", "AssumeQueueRoleForWorker"]),
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
    # The 500 name the current service sends. The model's name,
    # InternalServerErrorException, makes the agent exit (see
    # test_internal_server_error_is_retried), so it is not used here.
    st.tuples(
        st.sampled_from(["UpdateWorkerSchedule", "BatchGetJobEntity", "UpdateWorker"]),
        st.just("InternalServerException"),
        st.just({}),
        _ERROR_COUNT,
    ),
)
_ENTITY_ERROR_CODES = st.sampled_from(
    [
        "InternalServerException",
        "ValidationException",
        "ResourceNotFoundException",
        "MaxPayloadSizeExceeded",
    ]
)
_QUEUE_ROLE_ERRORS = st.sampled_from(
    [
        ("ResourceNotFoundException", {}),
        ("AccessDeniedException", {}),
        ("ConflictException", {"reason": "STATUS_CONFLICT", "resourceType": "queue"}),
        ("ValidationException", {}),
    ]
)
# Throttling on UpdateWorker during a drain is a known bug (the STOPPING timeout never
# starts), see test_worker_drain_finishes_when_update_worker_fails.
_UPDATE_WORKER_ERRORS = st.sampled_from(
    [
        None,
        ("ConflictException", {"reason": "STATUS_CONFLICT", "resourceType": "worker"}),
        ("ResourceNotFoundException", {}),
        ("AccessDeniedException", {}),
    ]
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
        self.worker_lost = False

    # ---- helpers -------------------------------------------------------

    @property
    def service(self) -> Any:
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

    def _open_queues(self) -> set[str]:
        return {self.service.sessions[session_id].queue_id for session_id in self._open_sessions()}

    def _queue_job(self, second: bool) -> tuple[str, str]:
        return (SECOND_QUEUE_ID, SECOND_JOB_ID) if second else (self.agent.queue_id, JOB_ID)

    # ---- rules: what the service does ----------------------------------

    @precondition(lambda self: not self.stopped and len(self._open_sessions()) < 3)
    @rule(
        second_queue=st.booleans(),
        environments=_ENVIRONMENTS,
        tasks=st.lists(_TASKS, max_size=3),
    )
    def assign_session(self, second_queue: bool, environments: list[str], tasks: list) -> None:
        # The service never gives one worker sessions from two queues at the same time.
        open_queues = self._open_queues()
        if open_queues:
            second_queue = SECOND_QUEUE_ID in open_queues
        queue_id, job_id = self._queue_job(second_queue)
        actions = [env_enter(name, job_id) for name in environments] + tasks
        self.session_ids.append(
            self.service.open_session(
                worker_id=self.agent.worker_id,
                queue_id=queue_id,
                job_id=job_id,
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
        self.agent.backend.inject_error(
            operation, code, count=count, resourceId=self.agent.worker_id, **fields
        )

    @precondition(lambda self: not self.stopped)
    @rule(
        second_queue=st.booleans(),
        kind=st.sampled_from(["jobDetails", "stepDetails", "environmentDetails"]),
        environment=st.sampled_from(ENVIRONMENT_NAMES),
        code=_ENTITY_ERROR_CODES,
        count=st.integers(min_value=1, max_value=10),
    )
    def return_entity_error(
        self, second_queue: bool, kind: str, environment: str, code: str, count: int
    ) -> None:
        _, job_id = self._queue_job(second_queue)
        entity_id = {
            "jobDetails": job_id,
            "stepDetails": STEP_ID,
            "environmentDetails": environment_id(environment, job_id),
        }[kind]
        self.service.inject_entity_error(
            kind=kind, job_id=job_id, entity_id=entity_id, code=code, count=count
        )

    @precondition(lambda self: not self.stopped)
    @rule(second_queue=st.booleans(), error=_QUEUE_ROLE_ERRORS)
    def return_queue_role_error(self, second_queue: bool, error: tuple) -> None:
        code, fields = error
        queue_id, _ = self._queue_job(second_queue)
        if code == "ConflictException":
            fields = {**fields, "resourceId": queue_id}
        self.service.inject_queue_role_error(queue_id, code, **fields)

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
    @rule(deleted=st.booleans())
    def service_loses_worker(self, deleted: bool) -> None:
        """The service marks the worker NOT_RESPONDING, or deletes it, and takes the work back."""
        if deleted:
            self.service.forget_worker(self.agent.worker_id)
            expected: type = DeadlineRequestWorkerNotFound
        else:
            self.service.mark_not_responding(self.agent.worker_id)
            expected = DeadlineRequestWorkerOfflineError
        outcome = self.agent.join(timeout=90)
        assert isinstance(outcome, expected), f"scheduler ended with {outcome!r}"
        self.stopped = True
        self.worker_lost = True

    @precondition(lambda self: not self.stopped)
    @rule(update_worker_error=_UPDATE_WORKER_ERRORS)
    def worker_drains_itself(self, update_worker_error: tuple | None) -> None:
        if update_worker_error is not None:
            code, fields = update_worker_error
            self.agent.backend.inject_error(
                "UpdateWorker", code, count=50, resourceId=self.agent.worker_id, **fields
            )
        self.agent.scheduler.shutdown(
            grace_time=timedelta(seconds=5), fail_message="The host is shutting down."
        )
        # A 5 second grace time must not take much longer than that.
        assert self.agent.join(timeout=30) is None
        self.agent.backend.injected_errors.pop("UpdateWorker", None)
        self.agent.set_status(WorkerStatus.STOPPED)
        self.stopped = True
        self.worker_drained = True

    @precondition(lambda self: self.stopped)
    @rule()
    def worker_is_gone(self) -> None:
        """Nothing is left to do after the worker stopped; the run ends at teardown."""

    # ---- checks --------------------------------------------------------

    @property
    def _ignored(self) -> frozenset[str]:
        return KNOWN_VIOLATIONS_AFTER_WORKER_DRAIN if self.worker_drained else KNOWN_VIOLATIONS

    @invariant()
    def no_protocol_violations(self) -> None:
        assert_no_violations(self.agent, ignore=self._ignored)

    @invariant()
    def credentials_belong_to_the_session_queue(self) -> None:
        assert credential_failures(self.agent) == []

    @invariant()
    def scheduler_is_running_until_stopped(self) -> None:
        if not self.stopped:
            assert self.agent.running, f"WorkerScheduler exited with {self.agent.outcome!r}"

    def teardown(self) -> None:
        try:
            if not self.stopped:
                assert self.agent.running, f"WorkerScheduler exited with {self.agent.outcome!r}"
                # Queued errors only slow the clean shutdown down; drop them.
                self.agent.backend.injected_errors.clear()
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
            if not self.worker_lost:
                for session_id in self.session_ids:
                    self.service.check_session_complete(
                        session_id, worker_drained=self.worker_drained
                    )
            assert_no_violations(self.agent, ignore=self._ignored)
            assert credential_failures(self.agent) == []
            # Killed processes can take a moment to leave the process table.
            deadline = time.monotonic() + 10
            while leftover_processes() and time.monotonic() < deadline:
                time.sleep(0.2)
            assert leftover_processes() == [], "the agent left script processes running"
        finally:
            self._stop_agent()
            self.monkeypatch.undo()
            shutil.rmtree(self.tmp_path, ignore_errors=True)


WorkerProtocolMachine.TestCase.settings = settings(
    max_examples=int(os.environ.get("WORKER_PROTOCOL_EXAMPLES", "10")),
    stateful_step_count=int(os.environ.get("WORKER_PROTOCOL_STEPS", "10")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    database=None,
    print_blob=True,
)

test_worker_protocol_state_machine = WorkerProtocolMachine.TestCase
