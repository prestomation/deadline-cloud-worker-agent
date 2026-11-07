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

| Test | Problem |
|---|---|
| `test_internal_server_error_is_retried` | The model names the 500 error `InternalServerErrorException`. The agent retries only `InternalServerException`, so it exits instead of retrying. |
| `test_env_exit_for_an_environment_never_entered_is_not_run` | After an `envEnter` fails, the agent runs the `envExit` of a later environment that was never entered, and reports it `FAILED`. The protocol says `NEVER_ATTEMPTED`. |

When a bug is fixed, its strict `xfail` test fails. Remove the marker, and remove the rule
from `KNOWN_VIOLATIONS` in the state machine test.
