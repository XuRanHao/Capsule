"""Trusted server-side routing for generic processing-task workers.

Task messages describe *what* to process, never a Python class, queue name, or
physical worker stream.  Worker deployment selects a route key locally and
uses this frozen registry to validate the message before invoking its processor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskMessage,
    ResourceClass,
)


class ProcessingTaskContractError(ValueError):
    """A delivery conflicts with the server-owned task routing contract."""


@dataclass(frozen=True, slots=True)
class ProcessorRegistration:
    """Immutable metadata selected by trusted worker deployment configuration."""

    task_kind: ProcessingTaskKind
    resource_class: ResourceClass
    route_key: str
    processor_version: int


def _frozen_routes() -> Mapping[
    ProcessingTaskKind,
    Mapping[ResourceClass, Mapping[str, Mapping[int, ProcessorRegistration]]],
]:
    video = ProcessorRegistration(
        task_kind=ProcessingTaskKind.VIDEO,
        resource_class=ResourceClass.MPS_VIDEO,
        route_key="mps_video",
        processor_version=1,
    )
    audio = ProcessorRegistration(
        task_kind=ProcessingTaskKind.AUDIO,
        resource_class=ResourceClass.MPS_VIDEO,
        route_key="mps_video",
        processor_version=1,
    )
    image = ProcessorRegistration(
        task_kind=ProcessingTaskKind.IMAGE,
        resource_class=ResourceClass.CPU,
        route_key="cpu_image",
        processor_version=1,
    )
    text = ProcessorRegistration(
        task_kind=ProcessingTaskKind.TEXT,
        resource_class=ResourceClass.CPU,
        route_key="cpu_text",
        processor_version=1,
    )
    return MappingProxyType(
        {
            ProcessingTaskKind.VIDEO: MappingProxyType(
                {
                    ResourceClass.MPS_VIDEO: MappingProxyType(
                        {"mps_video": MappingProxyType({1: video})}
                    )
                }
            ),
            ProcessingTaskKind.AUDIO: MappingProxyType(
                {
                    ResourceClass.MPS_VIDEO: MappingProxyType(
                        {"mps_video": MappingProxyType({1: audio})}
                    )
                }
            ),
            ProcessingTaskKind.IMAGE: MappingProxyType(
                {
                    ResourceClass.CPU: MappingProxyType(
                        {"cpu_image": MappingProxyType({1: image})}
                    )
                }
            ),
            ProcessingTaskKind.TEXT: MappingProxyType(
                {
                    ResourceClass.CPU: MappingProxyType(
                        {"cpu_text": MappingProxyType({1: text})}
                    )
                }
            ),
        }
    )


PROCESSOR_REGISTRY = _frozen_routes()


def resolve_processor_registration(
    *,
    task_kind: ProcessingTaskKind,
    resource_class: ResourceClass,
    route_key: str,
    processor_version: int,
) -> ProcessorRegistration:
    """Resolve only an exact server-approved kind/resource/route/version tuple."""
    try:
        resources = PROCESSOR_REGISTRY[task_kind]
        routes = resources[resource_class]
        versions = routes[route_key]
        return versions[processor_version]
    except KeyError as exc:
        raise ProcessingTaskContractError(
            "unsupported processing route "
            f"kind={task_kind.value} resource={resource_class.value} "
            f"route={route_key!r} version={processor_version}"
        ) from exc


def validate_message_contract(
    message: ProcessingTaskMessage,
    *,
    expected_route_key: str,
) -> ProcessorRegistration:
    """Reject task-kind/resource/version spoofing before processor dispatch.

    ``expected_route_key`` is worker-local deployment configuration.  The
    message route must equal it and an exact frozen registration must exist;
    neither task producers nor messages select a Python implementation or a
    physical queue/stream.
    """
    resource_class = message.resource_class
    if resource_class is None:  # pragma: no cover - message post-init sets it.
        raise ProcessingTaskContractError("processing task has no resource class")
    if message.message_schema_version != 1 or message.dispatch_version != 0:
        raise ProcessingTaskContractError(
            "unsupported processing message schema or dispatch version"
        )
    if message.route_key != expected_route_key:
        raise ProcessingTaskContractError(
            f"task route {message.route_key!r} does not match worker route {expected_route_key!r}"
        )
    return resolve_processor_registration(
        task_kind=message.task_kind,
        resource_class=resource_class,
        route_key=message.route_key,
        processor_version=message.processor_version,
    )
