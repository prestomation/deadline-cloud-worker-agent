# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Tests for DeadlineClient.supports_session_action_failure_reason and the
botocore serialization dependency it guards.

The worker may send ``updatedSessionActions.*.failureReason`` only when the
public SDK model it resolved declares the member. These tests pin the two
facts that make the capability check the release-ordering gate:

1. botocore parameter validation rejects an unknown request member BEFORE any
   request is sent, so emitting without model support would fail every
   UpdateWorkerSchedule call.
2. The capability check reads exactly the member botocore validates.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.exceptions import ParamValidationError
from botocore.model import ServiceModel
from botocore.stub import Stubber

from deadline_worker_agent.boto.shim import DeadlineClient


def _uws_service_description(*, with_failure_reason: bool) -> dict[str, Any]:
    """A minimal Deadline service description containing only
    UpdateWorkerSchedule, with or without the new member."""
    info_members: dict[str, Any] = {
        "completedStatus": {"shape": "CompletedStatus"},
        "progressMessage": {"shape": "String"},
    }
    shapes: dict[str, Any] = {
        "String": {"type": "string"},
        "CompletedStatus": {
            "type": "string",
            "enum": ["SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELED", "NEVER_ATTEMPTED"],
        },
        "UpdatedSessionActionInfo": {"type": "structure", "members": info_members},
        "UpdatedSessionActions": {
            "type": "map",
            "key": {"shape": "String"},
            "value": {"shape": "UpdatedSessionActionInfo"},
        },
        "UpdateWorkerScheduleRequest": {
            "type": "structure",
            "required": ["farmId", "fleetId", "workerId"],
            "members": {
                "farmId": {"shape": "String", "location": "uri", "locationName": "farmId"},
                "fleetId": {"shape": "String", "location": "uri", "locationName": "fleetId"},
                "workerId": {"shape": "String", "location": "uri", "locationName": "workerId"},
                "updatedSessionActions": {"shape": "UpdatedSessionActions"},
            },
        },
        "UpdateWorkerScheduleResponse": {
            "type": "structure",
            "members": {"updateIntervalSeconds": {"shape": "Integer"}},
        },
        "Integer": {"type": "integer"},
    }
    if with_failure_reason:
        shapes["SessionActionFailureReason"] = {"type": "string", "enum": ["ACTION_TIMEOUT"]}
        info_members["failureReason"] = {"shape": "SessionActionFailureReason"}
    return {
        "version": "2.0",
        "metadata": {
            "apiVersion": "2023-10-12",
            "endpointPrefix": "deadline",
            "protocol": "rest-json",
            "serviceFullName": "AWSDeadlineCloud",
            "serviceId": "deadline",
            "signatureVersion": "v4",
            "uid": "deadline-2023-10-12",
        },
        "operations": {
            "UpdateWorkerSchedule": {
                "name": "UpdateWorkerSchedule",
                "http": {
                    "method": "PATCH",
                    "requestUri": "/2023-10-12/farms/{farmId}/fleets/{fleetId}/workers/{workerId}/schedule",
                },
                "input": {"shape": "UpdateWorkerScheduleRequest"},
                "output": {"shape": "UpdateWorkerScheduleResponse"},
            }
        },
        "shapes": shapes,
    }


class _FakeMeta:
    def __init__(self, service_model: ServiceModel) -> None:
        self.service_model = service_model


class _FakeBotoClient:
    def __init__(self, service_model: ServiceModel) -> None:
        self.meta = _FakeMeta(service_model)


class TestSupportsSessionActionFailureReason:
    def test_true_when_model_declares_member(self) -> None:
        model = ServiceModel(_uws_service_description(with_failure_reason=True), "deadline")
        client = DeadlineClient(_FakeBotoClient(model))

        assert client.supports_session_action_failure_reason() is True

    def test_false_when_model_omits_member(self) -> None:
        model = ServiceModel(_uws_service_description(with_failure_reason=False), "deadline")
        client = DeadlineClient(_FakeBotoClient(model))

        assert client.supports_session_action_failure_reason() is False

    def test_false_for_uninspectable_client(self) -> None:
        # A mock client (the shim's own testing mode) has no real model.
        assert DeadlineClient(MagicMock()).supports_session_action_failure_reason() is False
        assert DeadlineClient(object()).supports_session_action_failure_reason() is False

    def test_matches_installed_public_sdk_model(self) -> None:
        """The check reads the installed botocore/boto3 'deadline' model.
        Whatever that model says today is what the worker will do."""
        session = boto3.session.Session(region_name="us-west-2")
        real_client = session.client(
            "deadline",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        members = (
            real_client.meta.service_model.operation_model("UpdateWorkerSchedule")
            .input_shape.members["updatedSessionActions"]
            .value.members
        )

        assert DeadlineClient(real_client).supports_session_action_failure_reason() is (
            "failureReason" in members
        )
        if "failureReason" in members:
            # Once published, the model must carry the closed enum the worker emits.
            assert "ACTION_TIMEOUT" in (members["failureReason"].enum or [])


class TestBotocoreSerializationDependency:
    """Pins the botocore behavior the gate exists for."""

    @staticmethod
    def _client_for(service_model: ServiceModel):
        session = boto3.session.Session(region_name="us-west-2")
        client = session.client(
            "deadline",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            endpoint_url="https://deadline.example.invalid",
        )
        # Swap in the minimal model so the test does not depend on the
        # installed SDK release. The serializer, validator, and stubber all
        # read the model through client.meta.service_model / _service_model.
        client.meta.service_model.__dict__.update(service_model.__dict__)
        return client

    def test_unknown_member_is_rejected_before_any_request(self) -> None:
        client = self._client_for(
            ServiceModel(_uws_service_description(with_failure_reason=False), "deadline")
        )

        def request_attempted(**kwargs: Any) -> None:  # pragma: no cover - failure path
            raise AssertionError("a request was attempted despite the invalid parameter")

        # before-call fires after parameter validation and before the HTTP
        # request; parameter validation must stop the call before it.
        client.meta.events.register("before-call.deadline.UpdateWorkerSchedule", request_attempted)
        with pytest.raises(ParamValidationError) as exc_info:
            client.update_worker_schedule(
                farmId="farm-1",
                fleetId="fleet-1",
                workerId="worker-1",
                updatedSessionActions={
                    "action-1": {
                        "completedStatus": "FAILED",
                        "failureReason": "ACTION_TIMEOUT",
                    }
                },
            )
        assert "failureReason" in str(exc_info.value)

    def test_declared_member_serializes(self) -> None:
        # Note: botocore validates member names and types, not enum values;
        # the closed ACTION_TIMEOUT vocabulary is enforced by the worker's
        # SessionActionFailureReason type and by the service.
        client = self._client_for(
            ServiceModel(_uws_service_description(with_failure_reason=True), "deadline")
        )
        with Stubber(client) as stubber:
            stubber.add_response(
                "update_worker_schedule",
                {"updateIntervalSeconds": 15},
                expected_params={
                    "farmId": "farm-1",
                    "fleetId": "fleet-1",
                    "workerId": "worker-1",
                    "updatedSessionActions": {
                        "action-1": {
                            "completedStatus": "FAILED",
                            "failureReason": "ACTION_TIMEOUT",
                        }
                    },
                },
            )
            response = client.update_worker_schedule(
                farmId="farm-1",
                fleetId="fleet-1",
                workerId="worker-1",
                updatedSessionActions={
                    "action-1": {"completedStatus": "FAILED", "failureReason": "ACTION_TIMEOUT"}
                },
            )
            assert response["updateIntervalSeconds"] == 15
            stubber.assert_no_pending_responses()
