"""Server-owned routing contracts for generic processing tasks."""

from __future__ import annotations

import pytest

from capsule.pipeline.processing_task_registry import (
    PROCESSOR_REGISTRY,
    ProcessingTaskContractError,
    resolve_processor_registration,
    validate_message_contract,
)
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskMessage,
    ResourceClass,
)


def _message(**updates: object) -> ProcessingTaskMessage:
    values: dict[str, object] = {
        "job_id": "job-1",
        "workspace_id": "workspace-1",
        "source_file_id": "source-1",
        "generation": 1,
    }
    values.update(updates)
    return ProcessingTaskMessage(**values)  # type: ignore[arg-type]


def test_video_mps_registration_is_exact_and_frozen() -> None:
    registration = validate_message_contract(_message(), expected_route_key="mps_video")

    assert registration.task_kind is ProcessingTaskKind.VIDEO
    assert registration.resource_class is ResourceClass.MPS_VIDEO
    assert registration.processor_version == 1
    with pytest.raises(TypeError):
        PROCESSOR_REGISTRY[ProcessingTaskKind.TEXT] = {}  # type: ignore[index]


@pytest.mark.parametrize(
    ("kind", "route_key"),
    [
        (ProcessingTaskKind.IMAGE, "cpu_image"),
        (ProcessingTaskKind.TEXT, "cpu_text"),
    ],
)
def test_cpu_registration_has_an_independent_trusted_worker_route(
    kind: ProcessingTaskKind,
    route_key: str,
) -> None:
    registration = validate_message_contract(
        _message(
            task_kind=kind,
            resource_class=ResourceClass.CPU,
            route_key=route_key,
        ),
        expected_route_key=route_key,
    )

    assert registration.task_kind is kind
    assert registration.resource_class is ResourceClass.CPU
    assert registration.route_key == route_key


@pytest.mark.parametrize(
    ("message", "expected_route_key"),
    [
        (
            _message(
                task_kind=ProcessingTaskKind.TEXT,
                resource_class=ResourceClass.CPU,
                route_key="cpu_image",
            ),
            "cpu_image",
        ),
        (
            _message(
                task_kind=ProcessingTaskKind.IMAGE,
                resource_class=ResourceClass.CPU,
                route_key="cpu_text",
            ),
            "cpu_text",
        ),
        (_message(resource_class=ResourceClass.CPU), "mps_video"),
        (_message(processor_version=2), "mps_video"),
        (_message(route_key="other_mps_video"), "mps_video"),
        (_message(message_schema_version=2), "mps_video"),
        (_message(dispatch_version=1), "mps_video"),
        (_message(), "untrusted-python-class"),
    ],
)
def test_registry_rejects_unknown_kind_and_spoofed_route_contract(
    message: ProcessingTaskMessage,
    expected_route_key: str,
) -> None:
    with pytest.raises(ProcessingTaskContractError):
        validate_message_contract(message, expected_route_key=expected_route_key)


def test_resolve_has_no_stream_or_python_processor_input() -> None:
    registration = resolve_processor_registration(
        task_kind=ProcessingTaskKind.VIDEO,
        resource_class=ResourceClass.MPS_VIDEO,
        route_key="mps_video",
        processor_version=1,
    )

    assert registration.route_key == "mps_video"
