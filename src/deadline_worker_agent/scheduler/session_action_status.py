# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from openjd.sessions import ActionStatus

if TYPE_CHECKING:
    from ..api_models import CompletedActionStatus, ManifestInfo


@dataclass(frozen=True)
class SessionActionStatus:
    id: str
    update_time: datetime | None = None
    status: ActionStatus | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    completed_status: CompletedActionStatus | None = None
    manifests: list[ManifestInfo] | None = None
    session_id: str | None = None
    """The ID of the session the action belongs to. Diagnostic metadata for
    the protocol trace; never sent to the service."""
    kind: str | None = None
    """The action kind (a SessionActionLogKind value or an API actionType
    value). Diagnostic metadata for the protocol trace; never sent to the
    service."""
