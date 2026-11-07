# Worker protocol tests

These tests run the real `WorkerScheduler` against a hermetic Deadline Cloud service
and check its behavior against [the worker API protocol](../../../docs/dev/worker_api_protocol.md).
They need no AWS account and no network access.

## How it works

- The service is `MockDeadlineBackend` from `deadline-cloud-test-fixtures`. It serves the
  worker APIs on `127.0.0.1` over REST-JSON, and validates each response against the
  botocore service model. Its `MockWorkerScheduler` holds the service-side state (workers,
  sessions, session actions) and checks every `UpdateWorkerSchedule` report against the
  protocol rules. A broken rule is recorded as a `ProtocolViolation`.
- The agent uses a real boto3 `deadline` client with the agent's own botocore config.
  Session actions run real Open Job Description sessions with short Python subprocesses.
- Only session logs (sent to a `NullHandler`) and telemetry (opted out) are replaced.

## Files

| File | What it does |
|---|---|
| `harness.py` | Starts the mock service and a `WorkerScheduler`, and defines the job, step and environments. |
| `test_worker_protocol_examples.py` | Fixed scenarios, one per protocol path. Known agent bugs are strict `xfail` tests. |
| `test_worker_protocol_state_machine.py` | A Hypothesis state machine that plays the service: assign, add, cancel, close, inject errors, drain. |

## Running

```sh
hatch run integ-test test/integ/protocol
# More and longer state machine runs:
WORKER_PROTOCOL_EXAMPLES=100 WORKER_PROTOCOL_STEPS=20 hatch run integ-test test/integ/protocol
```

The tests are skipped when the installed `deadline-cloud-test-fixtures` has no
`MockWorkerScheduler`.

## Known agent bugs

Each one is a strict `xfail` test in `test_worker_protocol_examples.py`.

| Test | Problem |
|---|---|
| `test_internal_server_error_is_retried[model-name]` | The model names the 500 error `InternalServerErrorException`. The agent retries only `InternalServerException` (the name the current service sends), so it exits on the model's name. |
| `test_env_exit_for_an_environment_never_entered_is_not_run` | If an `envExit` for an environment that was never entered is queued, the agent runs it and reports it `FAILED` instead of `NEVER_ATTEMPTED`. |
| `test_worker_drain_finishes_when_update_worker_fails[throttling]` | `_transition_to_stopping()` creates its timeout `Timer` but never starts it, so a throttled `UpdateWorker(STOPPING)` holds up the drain. |
| `test_worker_drain_reports_actions_of_a_session_that_is_still_starting` | The drain sends its last `UpdateWorkerSchedule` before session threads report their cancels, so those reports are lost. |
| `test_task_queued_before_a_setup_failure_is_not_run` | After a session fails during setup, the agent forgets it. A task the service queued before it saw the failure is then run in a new copy of the session. |
| `test_task_queued_after_a_failure_is_reported_never_attempted` | A task that reaches a session after a failure is reported `FAILED` instead of `NEVER_ATTEMPTED`. |

When a bug is fixed, its strict `xfail` test fails. Remove the marker, and remove the rule
from `KNOWN_VIOLATIONS` in the state machine test.

## What the fake service does not model

- The real scheduler rejects a non-`NEVER_ATTEMPTED` report after an unsuccessful action
  with a `ValidationException`. The fake service records a violation and keeps going.
- The real scheduler cancels already-assigned later actions after a failure. The fake
  service leaves them assigned until the worker reports them.
- Throttling limits, timeouts and the worker health tracker are not modeled. Tests inject
  errors and state changes directly.
