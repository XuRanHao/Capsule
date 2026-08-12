"""Redis integration coverage for processing-task quarantine semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from capsule.config import Settings
from capsule.pipeline.video_task_runtime import (
    RedisVideoTaskQueue,
    VideoTaskDelivery,
    VideoTaskMessage,
    VideoTaskRuntime,
)

_LEGACY_FIELDS = {
    "task_id": "video-dd2bf748b62ebf2b995c870f",
    "job_id": "job-legacy",
    "workspace_id": "workspace-legacy",
    "source_file_id": "source-legacy",
    "generation": "3",
    "attempt": "0",
    "result_version": "2",
    "source_uri": "file:///imports/legacy.mp4",
}


@dataclass(frozen=True, slots=True)
class _RedisTestResources:
    admin: Redis
    stream: str
    group: str
    dlq_stream: str
    quarantine_stream: str
    delayed_key: str
    consumer: str

    async def cleanup(self) -> None:
        try:
            await self.admin.delete(
                self.stream,
                self.dlq_stream,
                self.quarantine_stream,
                self.delayed_key,
            )
        finally:
            await self.admin.aclose()


async def _redis_test_resources() -> _RedisTestResources:
    settings = Settings()
    admin = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await admin.ping()
    except RedisError:
        await admin.aclose()
        pytest.skip("Redis integration service is unavailable")

    suffix = uuid4().hex
    prefix = f"capsule:test:processing-task:{suffix}"
    return _RedisTestResources(
        admin=admin,
        stream=f"{prefix}:stream",
        group=f"processing-task-test-{suffix}",
        dlq_stream=f"{prefix}:dlq",
        quarantine_stream=f"{prefix}:quarantine",
        delayed_key=f"{prefix}:delayed",
        consumer=f"consumer-{suffix}",
    )


def _queue(resources: _RedisTestResources) -> RedisVideoTaskQueue:
    return RedisVideoTaskQueue(
        redis_url=Settings().redis_url,
        stream=resources.stream,
        group=resources.group,
        consumer=resources.consumer,
        dlq_stream=resources.dlq_stream,
        quarantine_stream=resources.quarantine_stream,
        delayed_key=resources.delayed_key,
        # The test receives consecutive entries before ACKing them, so no PEL
        # entry should be reclaimed between receives.
        claim_idle_ms=60_000,
    )


@pytest.mark.integration
async def test_real_redis_quarantines_malformed_entry_then_receives_legacy_and_current() -> None:
    resources = await _redis_test_resources()
    queue = _queue(resources)
    legacy = VideoTaskMessage.from_fields(_LEGACY_FIELDS)
    current = VideoTaskMessage(
        job_id="job-current",
        workspace_id="workspace-current",
        source_file_id="source-current",
        generation=1,
        source_uri="file:///imports/current.mp4",
    )
    try:
        await queue.start()
        malformed_receipt = await resources.admin.xadd(
            resources.stream,
            {
                "task_id": "malformed",
                "route_key": "mps_video",
                "source_uri": "file:///private/imports/secret.mp4",
            },
        )
        legacy_receipt = await resources.admin.xadd(resources.stream, _LEGACY_FIELDS)
        await queue.publish(current)

        received_legacy = await queue.receive()
        received_current = await queue.receive()

        assert received_legacy == VideoTaskDelivery(message=legacy, receipt=legacy_receipt)
        assert received_legacy.message.legacy_wire
        assert received_current.message == current

        quarantine_entries = await resources.admin.xrange(resources.quarantine_stream)
        assert len(quarantine_entries) == 1
        _, quarantined = quarantine_entries[0]
        assert quarantined["source_receipt"] == malformed_receipt
        assert quarantined["quarantine_reason"] == "invalid_task_contract"
        assert "source_uri" not in quarantined

        pending = await resources.admin.xpending_range(
            resources.stream,
            resources.group,
            min="-",
            max="+",
            count=10,
        )
        assert malformed_receipt not in {entry["message_id"] for entry in pending}

        await queue.acknowledge(received_legacy)
        await queue.acknowledge(received_current)
        assert (await resources.admin.xpending(resources.stream, resources.group))["pending"] == 0
    finally:
        try:
            await queue.close()
        finally:
            await resources.cleanup()


class _RouteSpoofRepository:
    def __init__(self) -> None:
        self.inspect_calls = 0
        self.claim_calls = 0

    async def inspect_message_contract(self, _message: VideoTaskMessage) -> bool | None:
        self.inspect_calls += 1
        raise AssertionError("route validation must precede durable contract inspection")

    async def claim_attempt(self, *_args: Any, **_kwargs: Any) -> None:
        self.claim_calls += 1
        raise AssertionError("route spoof must not be claimed")


@pytest.mark.integration
async def test_real_redis_route_spoof_quarantine_is_published_before_delivery_ack() -> None:
    resources = await _redis_test_resources()
    queue = _queue(resources)
    repository = _RouteSpoofRepository()
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=cast(Any, repository),
        processor=cast(Any, object()),
        worker_id="integration-worker",
    )
    spoofed = VideoTaskMessage(
        job_id="job-spoofed",
        workspace_id="workspace-spoofed",
        source_file_id="source-spoofed",
        generation=1,
        route_key="untrusted-route",
    )
    try:
        await queue.start()
        await queue.publish(spoofed)
        delivery = await queue.receive()

        outcome = await runtime.handle_delivery(delivery)

        assert outcome == "quarantined"
        assert repository.inspect_calls == 0
        assert repository.claim_calls == 0
        quarantine_entries = await resources.admin.xrange(resources.quarantine_stream)
        assert len(quarantine_entries) == 1
        _, quarantined = quarantine_entries[0]
        assert quarantined["source_receipt"] == delivery.receipt
        assert quarantined["error"] == (
            "processing task contract rejected: ProcessingTaskContractError"
        )
        assert (await resources.admin.xpending(resources.stream, resources.group))["pending"] == 0
    finally:
        try:
            await queue.close()
        finally:
            await resources.cleanup()
