# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Structured protocol trace for the Worker Agent.

This module implements an opt-in, default-off, structured trace of the
Worker Agent's protocol-level lifecycle (process, worker loop, drain,
service sync, sessions, actions, and cancels). It is intended for external
tooling that observes the agent's behavior; it is NOT a substitute for the
agent's structured logs.

Configuration is via environment variables that are read once at process
startup and then removed from the process environment so that they are never
inherited by session action subprocesses:

- ``DEADLINE_WORKER_PROTOCOL_TRACE``: Off when unset or set to one of
  ``""``, ``0``, ``false``, ``off``, ``no`` (case-insensitive). Set to one of
  ``1``, ``true``, ``on``, ``yes``, ``stderr`` to emit to stderr. Any other
  value is treated as a file path; trace records are appended to that file.
- ``DEADLINE_WORKER_PROTOCOL_TRACE_STRICT``: Off by default. When enabled
  (``1``, ``true``, ``on``, ``yes``), programming errors (unknown events,
  disallowed payload keys or value types) and write failures raise instead of
  being silently dropped. Strict mode is for tests and development only.

Each record is a single line of JSON (an "envelope", canonical version 1)
with fields:

- ``v``: envelope format version (the integer 1)
- ``ts``: ISO-8601 UTC timestamp
- ``seq``: monotonically increasing sequence number (1-based, per process)
- ``src``: constant record source (``worker-agent``)
- ``event``: the event name; one of the allowlisted names in ``TraceEvent``
- ``run``: opaque unique ID for this agent process run
- ``corr``: optional correlation ID (for example a session or action ID)
- ``payload``: event payload; keys are allowlisted per event and values are
  restricted to scalars

The emitter is production-safe by design: when disabled every call is a
cheap no-op; when enabled, payloads are filtered against a per-event key
allowlist, values are restricted to scalar types with a bounded string
length, and (outside of strict mode) all emitter errors are swallowed. The
emitter disables itself after repeated consecutive write failures.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import IO, Any, Mapping, MutableMapping, Optional, Union

__all__ = [
    "PROTOCOL_TRACE_ENV_VAR",
    "PROTOCOL_TRACE_STRICT_ENV_VAR",
    "PROTOCOL_TRACE_ENV_VARS",
    "TraceEvent",
    "WorkerProtocolTrace",
    "emit",
    "get_trace",
    "initialize_from_environment",
    "scrub_protocol_trace_env_vars",
    "set_trace",
    "shutdown_trace",
]

PROTOCOL_TRACE_ENV_VAR = "DEADLINE_WORKER_PROTOCOL_TRACE"
PROTOCOL_TRACE_STRICT_ENV_VAR = "DEADLINE_WORKER_PROTOCOL_TRACE_STRICT"
PROTOCOL_TRACE_ENV_VARS = (
    PROTOCOL_TRACE_ENV_VAR,
    PROTOCOL_TRACE_STRICT_ENV_VAR,
)

ENVELOPE_VERSION = 1
ENVELOPE_SOURCE = "worker-agent"

_FALSE_VALUES = frozenset(("", "0", "false", "off", "no"))
_TRUE_VALUES = frozenset(("1", "true", "on", "yes"))
_STDERR_VALUES = _TRUE_VALUES | frozenset(("stderr",))

_MAX_STRING_LENGTH = 512
_MAX_CONSECUTIVE_WRITE_FAILURES = 5

_ScalarValue = Union[str, int, float, bool, None]


class TraceEvent(str, Enum):
    """The allowlisted protocol trace event names.

    Only these events may be emitted. Each event has an allowlisted set of
    payload keys (see ``_EVENT_PAYLOAD_KEYS``).
    """

    PROCESS_START = "process.start"
    PROCESS_STOP = "process.stop"
    WORKER_START = "worker.start"
    WORKER_STOP = "worker.stop"
    DRAIN_REQUESTED = "drain.requested"
    DRAIN_START = "drain.start"
    DRAIN_COMPLETE = "drain.complete"
    SYNC_START = "sync.start"
    SYNC_COMPLETE = "sync.complete"
    ACTION_REPORT = "action.report"
    SESSION_START = "session.start"
    SESSION_COMPLETE = "session.complete"
    SESSION_FAIL = "session.fail"
    ACTION_ASSIGNED = "action.assigned"
    ACTION_START = "action.start"
    ACTION_COMPLETE = "action.complete"
    ACTION_CANCEL = "action.cancel"


_EVENT_PAYLOAD_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        TraceEvent.PROCESS_START.value: frozenset(("agent_version", "platform")),
        TraceEvent.PROCESS_STOP.value: frozenset(),
        TraceEvent.WORKER_START.value: frozenset(("worker_id",)),
        TraceEvent.WORKER_STOP.value: frozenset(),
        TraceEvent.DRAIN_REQUESTED.value: frozenset(("grace_seconds",)),
        TraceEvent.DRAIN_START.value: frozenset(("session_count",)),
        TraceEvent.DRAIN_COMPLETE.value: frozenset(),
        TraceEvent.SYNC_START.value: frozenset(("updated_action_count", "interruptable")),
        TraceEvent.SYNC_COMPLETE.value: frozenset(
            (
                "assigned_session_count",
                "cancel_session_count",
                "update_interval_seconds",
                "desired_status",
            )
        ),
        TraceEvent.ACTION_REPORT.value: frozenset(
            ("status", "progress", "session_id", "has_timestamps")
        ),
        TraceEvent.SESSION_START.value: frozenset(("queue_id", "job_id", "worker_id")),
        TraceEvent.SESSION_COMPLETE.value: frozenset(("queue_id", "job_id")),
        TraceEvent.SESSION_FAIL.value: frozenset(("action_count",)),
        TraceEvent.ACTION_ASSIGNED.value: frozenset(("session_id", "kind", "env_id", "worker_id")),
        TraceEvent.ACTION_START.value: frozenset(("session_id", "kind", "env_id")),
        TraceEvent.ACTION_COMPLETE.value: frozenset(("status", "session_id", "kind")),
        TraceEvent.ACTION_CANCEL.value: frozenset(("session_id",)),
    }
)


class WorkerProtocolTrace:
    """A structured protocol trace emitter.

    Instances are safe for concurrent use from multiple threads. A disabled
    instance turns every ``emit()`` call into a cheap no-op.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        stream: Optional[IO[str]] = None,
        file_path: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        enabled : bool
            Whether the trace is enabled. When False, emit() is a no-op.
        stream : IO[str] | None
            The text stream to write records to. Mutually exclusive with
            file_path. Defaults to sys.stderr when enabled and file_path is
            not given.
        file_path : str | None
            A file to append records to. The file is opened lazily on first
            emit.
        strict : bool
            When True, emitter errors raise instead of being swallowed.
        """
        self._enabled = enabled
        self._strict = strict
        self._stream = stream
        self._file_path = file_path
        self._file: Optional[IO[str]] = None
        self._lock = threading.Lock()
        self._seq = 0
        self._run_id = uuid.uuid4().hex
        self._consecutive_write_failures = 0

    @property
    def enabled(self) -> bool:
        """Whether this trace emits records."""
        return self._enabled

    @property
    def strict(self) -> bool:
        """Whether emitter errors raise instead of being swallowed."""
        return self._strict

    @property
    def run_id(self) -> str:
        """The opaque unique ID for this agent process run."""
        return self._run_id

    def emit(
        self,
        event: Union[TraceEvent, str],
        *,
        corr: Optional[str] = None,
        **payload: _ScalarValue,
    ) -> None:
        """Emit a single trace record.

        This never raises unless strict mode is enabled.

        Parameters
        ----------
        event : TraceEvent | str
            The event name. Must be one of the allowlisted event names.
        corr : str | None
            An optional correlation ID (for example a session or action ID).
        **payload
            Scalar payload values. Keys not in the event's allowlist are
            dropped (strict mode: raise).
        """
        if not self._enabled:
            return
        try:
            self._emit_impl(event, corr=corr, payload=payload)
        except Exception:
            if self._strict:
                raise
            # Production-safe: never let tracing break the agent.

    def _emit_impl(
        self,
        event: Union[TraceEvent, str],
        *,
        corr: Optional[str],
        payload: dict[str, _ScalarValue],
    ) -> None:
        event_name = event.value if isinstance(event, TraceEvent) else str(event)
        allowed_keys = _EVENT_PAYLOAD_KEYS.get(event_name)
        if allowed_keys is None:
            if self._strict:
                raise ValueError(f"Unknown protocol trace event: {event_name!r}")
            return

        clean_payload: dict[str, _ScalarValue] = {}
        for key, value in payload.items():
            if key not in allowed_keys:
                if self._strict:
                    raise ValueError(
                        f"Payload key {key!r} is not allowlisted for event {event_name!r}"
                    )
                continue
            if not self._is_scalar(value):
                if self._strict:
                    raise TypeError(
                        f"Payload value for key {key!r} must be a scalar; got {type(value).__name__}"
                    )
                continue
            if isinstance(value, str) and len(value) > _MAX_STRING_LENGTH:
                value = value[:_MAX_STRING_LENGTH]
            clean_payload[key] = value

        with self._lock:
            self._seq += 1
            record: dict[str, Any] = {
                "v": ENVELOPE_VERSION,
                "ts": datetime.now(timezone.utc).isoformat(),
                "seq": self._seq,
                "src": ENVELOPE_SOURCE,
                "event": event_name,
                "run": self._run_id,
                "corr": corr,
                "payload": clean_payload,
            }
            line = json.dumps(record, separators=(",", ":"), default=str)
            self._write_line(line)

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    def _write_line(self, line: str) -> None:
        # Called with self._lock held.
        try:
            stream = self._get_stream()
            stream.write(line + "\n")
            stream.flush()
        except Exception:
            self._consecutive_write_failures += 1
            if self._consecutive_write_failures >= _MAX_CONSECUTIVE_WRITE_FAILURES:
                # Self-disable so that a persistently broken destination
                # cannot degrade the agent.
                self._enabled = False
            raise
        else:
            self._consecutive_write_failures = 0

    def _get_stream(self) -> IO[str]:
        if self._stream is not None:
            return self._stream
        if self._file_path is not None:
            if self._file is None:
                self._file = open(self._file_path, "a", encoding="utf-8")
            return self._file
        return sys.stderr

    def close(self) -> None:
        """Flush and close any file opened by this trace. Never raises."""
        with self._lock:
            self._enabled = False
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None


_trace_lock = threading.Lock()
_trace: WorkerProtocolTrace = WorkerProtocolTrace(enabled=False)


def get_trace() -> WorkerProtocolTrace:
    """Return the process-wide trace instance."""
    return _trace


def set_trace(trace: WorkerProtocolTrace) -> WorkerProtocolTrace:
    """Replace the process-wide trace instance and return the previous one.

    Primarily intended for tests.
    """
    global _trace
    with _trace_lock:
        previous = _trace
        _trace = trace
    return previous


def emit(
    event: Union[TraceEvent, str],
    *,
    corr: Optional[str] = None,
    **payload: _ScalarValue,
) -> None:
    """Emit a record through the process-wide trace instance (see
    WorkerProtocolTrace.emit)."""
    _trace.emit(event, corr=corr, **payload)


def shutdown_trace() -> None:
    """Close the process-wide trace instance. Never raises."""
    _trace.close()


def scrub_protocol_trace_env_vars(env: MutableMapping[str, str]) -> None:
    """Remove the protocol trace environment variables from a mapping.

    Used to guarantee that the trace settings never reach session action
    subprocesses (or any other child process) through their environment.
    """
    for var in PROTOCOL_TRACE_ENV_VARS:
        env.pop(var, None)


def initialize_from_environment(
    environ: Optional[MutableMapping[str, str]] = None,
) -> WorkerProtocolTrace:
    """Initialize the process-wide trace from environment variables.

    Reads ``DEADLINE_WORKER_PROTOCOL_TRACE`` and
    ``DEADLINE_WORKER_PROTOCOL_TRACE_STRICT``, then removes both from the
    environment so that no child process (including session action
    subprocesses) can inherit them.

    Parameters
    ----------
    environ : MutableMapping[str, str] | None
        The environment mapping to read and scrub. Defaults to ``os.environ``.

    Returns
    -------
    WorkerProtocolTrace
        The newly installed process-wide trace instance.
    """
    if environ is None:
        environ = os.environ

    raw_value = environ.get(PROTOCOL_TRACE_ENV_VAR, "")
    raw_strict = environ.get(PROTOCOL_TRACE_STRICT_ENV_VAR, "")
    scrub_protocol_trace_env_vars(environ)

    strict = raw_strict.strip().lower() in _TRUE_VALUES
    value = raw_value.strip()
    lowered = value.lower()

    trace: WorkerProtocolTrace
    if lowered in _FALSE_VALUES:
        trace = WorkerProtocolTrace(enabled=False, strict=strict)
    elif lowered in _STDERR_VALUES:
        trace = WorkerProtocolTrace(enabled=True, strict=strict)
    else:
        # Any other value is treated as a file path to append to.
        trace = WorkerProtocolTrace(enabled=True, file_path=value, strict=strict)

    set_trace(trace)
    return trace
