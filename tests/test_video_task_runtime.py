"""Reliability contract for the durable whole-video task runtime.

These tests deliberately use fakes: correctness of the state-machine must not
depend on a locally running Redis or PostgreSQL service.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskLease,
    ProcessingTaskMessage,
    ProcessingTaskProcessor,
    RedisVideoTaskQueue,
    ResourceClass,
    RetryPolicy,
    VideoTaskDelivery,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskRecoveryScheduler,
    VideoTaskResult,
    VideoTaskRuntime,
)

_LEGACY_VIDEO_FIELDS = {
    "task_id": "video-dd2bf748b62ebf2b995c870f",
    "job_id": "job-1",
    "workspace_id": "workspace-1",
    "source_file_id": "source-1",
    "generation": "3",
    "attempt": "0",
    "result_version": "2",
    "source_uri": "file:///imports/demo.mp4",
}


def _legacy_video_decoder(fields: dict[str, str]) -> dict[str, str]:
    """Frozen pre-generalization decoder: ignore every unrecognized field."""
    return {key: fields[key] for key in _LEGACY_VIDEO_FIELDS}


def test_generic_message_serializes_routing_and_keeps_legacy_video_defaults() -> None:
    legacy = ProcessingTaskMessage.from_fields(_LEGACY_VIDEO_FIELDS)
    legacy_fields = legacy.to_fields()
    restored = ProcessingTaskMessage.from_fields(_LEGACY_VIDEO_FIELDS)
    text = ProcessingTaskMessage(
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        task_kind=ProcessingTaskKind.TEXT,
        processor_version=4,
    )

    assert legacy.task_kind is ProcessingTaskKind.VIDEO
    assert legacy.processor_version == 1
    assert legacy.resource_class is ResourceClass.MPS_VIDEO
    assert restored == legacy
    assert legacy.task_id == _LEGACY_VIDEO_FIELDS["task_id"]
    assert _legacy_video_decoder(legacy_fields) == _LEGACY_VIDEO_FIELDS
    assert legacy_fields["message_schema_version"] == "1"
    assert legacy_fields["dispatch_version"] == "0"
    assert legacy_fields["route_key"] == "mps_video"
    assert text.resource_class is ResourceClass.CPU
    assert text.task_id != legacy.task_id
    assert text.to_fields()["task_kind"] == "text"
    assert text.to_fields()["processor_version"] == "4"
    assert text.to_fields()["resource_class"] == "cpu"
    assert VideoTaskMessage is ProcessingTaskMessage
    assert VideoTaskProgress.__name__ == "ProcessingTaskProgress"
    assert ProcessingTaskProcessor is not None


@pytest.mark.parametrize(
    "field",
    [
        "task_kind",
        "processor_version",
        "resource_class",
        "message_schema_version",
        "dispatch_version",
        "route_key",
    ],
)
def test_partial_contract_payload_is_not_treated_as_legacy(field: str) -> None:
    partial = dict(_LEGACY_VIDEO_FIELDS)
    partial[field] = "video" if field == "task_kind" else "1"

    with pytest.raises(ValueError, match="partial routing contract"):
        ProcessingTaskMessage.from_fields(partial)


def test_nonlegacy_task_id_matches_source_generation_kind_and_result_version() -> None:
    first = ProcessingTaskMessage(
        job_id="job-a",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        task_kind=ProcessingTaskKind.TEXT,
        result_version=2,
    )
    retried_job = ProcessingTaskMessage(
        job_id="job-b",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        task_kind=ProcessingTaskKind.TEXT,
        result_version=2,
    )
    new_result = ProcessingTaskMessage(
        job_id="job-a",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        task_kind=ProcessingTaskKind.TEXT,
        result_version=3,
    )
    lease = ProcessingTaskLease(
        task_id=first.task_id or "",
        source_file_id="source-1",
        source_generation=3,
        attempt=1,
        worker_id="worker-a",
        result_version=2,
        task_kind=ProcessingTaskKind.TEXT,
        resource_class=ResourceClass.CPU,
        route_key="text_cpu",
        processor_version=2,
        lease_token="database-fence",
    )

    assert first.task_id == retried_job.task_id
    assert first.task_id != new_result.task_id
    assert lease.task_kind is ProcessingTaskKind.TEXT
    assert lease.resource_class is ResourceClass.CPU
    assert lease.route_key == "text_cpu"
    assert lease.processor_version == 2
    assert lease.lease_token == "database-fence"


def _message(*, attempt: int = 0) -> VideoTaskMessage:
    return VideoTaskMessage(
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        attempt=attempt,
        result_version=2,
        source_uri="file:///imports/demo.mp4",
    )


class _FakeRedis:
    def __init__(self) -> None:
        self.claimed: list[tuple[str, dict[str, str]]] = []
        self.added: list[tuple[str, dict[str, str]]] = []
        self.acked: list[tuple[str, str, str]] = []
        self.new_messages: list[tuple[str, dict[str, str]]] = []
        self.closed = False

    async def xgroup_create(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def xadd(self, stream: str, fields: dict[str, str]) -> str:
        self.added.append((stream, fields))
        return "20-0"

    async def xautoclaim(self, *_args: Any, **_kwargs: Any) -> tuple[str, list[Any], list[Any]]:
        claimed, self.claimed = self.claimed, []
        return ("0-0", claimed, [])

    async def xreadgroup(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        messages, self.new_messages = self.new_messages, []
        return [("capsule:video:tasks", messages)] if messages else []

    async def zrangebyscore(self, *_args: Any, **_kwargs: Any) -> list[str]:
        return []

    async def xack(self, stream: str, group: str, receipt: str) -> int:
        self.acked.append((stream, group, receipt))
        return 1

    async def aclose(self) -> None:
        self.closed = True


@dataclass
class _FakeRepository:
    events: list[str]
    claim: VideoTaskLease | None
    complete_result: bool = True
    progress_result: bool = True
    ack_unclaimed: bool = True
    contract_state: bool | None = True
    inspect_error: Exception | None = None
    inspected: list[VideoTaskMessage] = field(default_factory=list)
    retries: list[tuple[VideoTaskLease, str, float]] = field(default_factory=list)
    failures: list[tuple[VideoTaskLease, str]] = field(default_factory=list)

    async def inspect_message_contract(self, message: VideoTaskMessage) -> bool | None:
        self.inspected.append(message)
        if self.inspect_error is not None:
            raise self.inspect_error
        return self.contract_state

    async def claim_attempt(
        self,
        message: VideoTaskMessage,
        *,
        worker_id: str,
        receipt: str,
    ) -> VideoTaskLease | None:
        self.events.append("claim")
        assert worker_id == "worker-a"
        assert receipt == "11-0"
        return self.claim

    async def can_ack_unclaimed(self, _message: VideoTaskMessage) -> bool:
        return self.ack_unclaimed

    async def heartbeat(self, lease: VideoTaskLease) -> bool:
        self.events.append("heartbeat")
        return True

    async def record_progress(self, lease: VideoTaskLease, progress: VideoTaskProgress) -> bool:
        self.events.append("progress")
        assert progress == VideoTaskProgress(
            completed_units=12,
            detail={"out_time_us": 400_000},
        )
        return self.progress_result

    async def complete(self, lease: VideoTaskLease, result: VideoTaskResult) -> bool:
        self.events.append("commit")
        assert result.result_ref == "s3://capsule/video/manifest.json"
        return self.complete_result

    async def schedule_retry(self, lease: VideoTaskLease, error: str, retry_at: float) -> bool:
        self.events.append("schedule_retry")
        self.retries.append((lease, error, retry_at))
        return True

    async def fail(self, lease: VideoTaskLease, error: str) -> bool:
        self.events.append("fail")
        self.failures.append((lease, error))
        return True

    async def mark_dlq_published(self, _message: VideoTaskMessage) -> None:
        self.events.append("mark_dlq")


@dataclass
class _FakeQueue:
    events: list[str]
    acknowledgements: list[VideoTaskDelivery] = field(default_factory=list)
    retried: list[VideoTaskDelivery] = field(default_factory=list)
    dlq: list[tuple[VideoTaskDelivery, str]] = field(default_factory=list)
    quarantined: list[tuple[VideoTaskDelivery, str]] = field(default_factory=list)
    quarantine_error: Exception | None = None

    async def acknowledge(self, delivery: VideoTaskDelivery) -> None:
        self.events.append("ack")
        self.acknowledgements.append(delivery)

    async def retry(
        self,
        delivery: VideoTaskDelivery,
        *,
        next_attempt: int,
        delay_seconds: float,
    ) -> None:
        self.events.append("retry")
        assert next_attempt == 2
        assert delay_seconds == 5.0
        self.retried.append(delivery)

    async def route_dlq(self, delivery: VideoTaskDelivery, *, error: str) -> None:
        self.events.append("dlq")
        self.dlq.append((delivery, error))

    async def quarantine(self, delivery: VideoTaskDelivery, *, error: str) -> None:
        self.events.append("quarantine")
        if self.quarantine_error is not None:
            raise self.quarantine_error
        self.quarantined.append((delivery, error))
        await self.acknowledge(delivery)


class _SuccessfulProcessor:
    async def process(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        report_progress: Any,
    ) -> VideoTaskResult:
        assert message.generation == 3
        assert lease.attempt == 1
        # Let the runtime heartbeat task run before the processor reports progress.
        await asyncio.sleep(0)
        await report_progress(
            VideoTaskProgress(
                completed_units=12,
                detail={"out_time_us": 400_000},
            )
        )
        return VideoTaskResult(result_ref="s3://capsule/video/manifest.json")


class _FailingProcessor:
    async def process(self, *_args: Any, **_kwargs: Any) -> VideoTaskResult:
        raise RuntimeError("ffmpeg progress timeout")


def _delivery(*, attempt: int = 0) -> VideoTaskDelivery:
    return VideoTaskDelivery(message=_message(attempt=attempt), receipt="11-0")


def _lease(*, attempt: int = 1) -> VideoTaskLease:
    return VideoTaskLease(
        task_id=_message().task_id,
        source_file_id="source-1",
        source_generation=3,
        attempt=attempt,
        worker_id="worker-a",
        result_version=2,
        lease_token="test-claim-token",
    )


async def test_queue_message_contract_and_pel_recovery_precede_new_reads() -> None:
    client = _FakeRedis()
    message = _message(attempt=2)
    client.claimed = [("10-0", message.to_fields())]
    queue = RedisVideoTaskQueue(
        redis_url="redis://unused",
        stream="capsule:video:tasks",
        group="video-workers",
        consumer="worker-b",
        dlq_stream="capsule:video:dlq",
        claim_idle_ms=100,
        client=client,
    )

    await queue.start()
    await queue.publish(message)
    recovered = await queue.receive()

    assert client.added == [("capsule:video:tasks", message.to_fields())]
    assert recovered == VideoTaskDelivery(message=message, receipt="10-0")
    assert VideoTaskMessage.from_fields(message.to_fields()) == message
    assert message.task_id == _message(attempt=99).task_id
    await queue.close()
    assert client.closed


async def test_malformed_pel_entry_is_quarantined_before_later_valid_delivery() -> None:
    client = _FakeRedis()
    client.claimed = [
        (
            "10-0",
            {
                "task_id": "invalid",
                "source_uri": "file:///private/imports/secret.mp4",
                "route_key": "mps_video",
            },
        )
    ]
    client.new_messages = [("11-0", _message().to_fields())]
    queue = RedisVideoTaskQueue(
        redis_url="redis://unused",
        stream="capsule:video:tasks",
        group="video-workers",
        consumer="worker-b",
        dlq_stream="capsule:video:dlq",
        client=client,
    )

    await queue.start()
    delivery = await asyncio.wait_for(queue.receive(), timeout=0.5)

    assert delivery == _delivery()
    assert client.acked == [("capsule:video:tasks", "video-workers", "10-0")]
    quarantine_stream, quarantined = client.added[0]
    assert quarantine_stream == "capsule:video:tasks:quarantine"
    assert quarantined["source_receipt"] == "10-0"
    assert quarantined["quarantine_reason"] == "invalid_task_contract"
    assert "source_uri" not in quarantined


async def test_runtime_commits_database_before_acknowledging_stream_delivery() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease())
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = _delivery()

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "completed"
    assert "heartbeat" in events
    assert [event for event in events if event != "heartbeat"] == [
        "claim",
        "progress",
        "commit",
        "ack",
    ]
    assert queue.acknowledgements == [delivery]


async def test_spoofed_route_is_quarantined_without_claiming_or_failing_task() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease())
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = VideoTaskDelivery(
        message=VideoTaskMessage(
            job_id="job-1",
            workspace_id="workspace-1",
            source_file_id="source-1",
            generation=3,
            result_version=2,
            source_uri="file:///imports/demo.mp4",
            route_key="untrusted-route",
        ),
        receipt="11-0",
    )

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "quarantined"
    assert events == ["quarantine", "ack"]
    assert repository.failures == []
    assert queue.acknowledgements == [delivery]
    assert queue.quarantined[0][1] == (
        "processing task contract rejected: ProcessingTaskContractError"
    )


async def test_quarantine_write_failure_leaves_spoofed_delivery_pending() -> None:
    events: list[str] = []
    queue = _FakeQueue(events, quarantine_error=RuntimeError("quarantine unavailable"))
    repository = _FakeRepository(events, claim=_lease())
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = VideoTaskDelivery(
        message=VideoTaskMessage(
            job_id="job-1",
            workspace_id="workspace-1",
            source_file_id="source-1",
            generation=3,
            route_key="untrusted-route",
        ),
        receipt="11-0",
    )

    with pytest.raises(RuntimeError, match="quarantine unavailable"):
        await runtime.handle_delivery(delivery)

    assert events == ["quarantine"]
    assert repository.failures == []
    assert queue.acknowledgements == []


async def test_durable_workspace_or_source_spoof_is_quarantined_before_claim() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease(), contract_state=False)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = VideoTaskDelivery(
        message=VideoTaskMessage(
            job_id="job-foreign",
            workspace_id="workspace-foreign",
            source_file_id="source-foreign",
            generation=3,
        ),
        receipt="11-0",
    )

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "quarantined"
    assert repository.inspected == [delivery.message]
    assert events == ["quarantine", "ack"]
    assert queue.quarantined == [(delivery, "durable_contract_mismatch")]
    assert repository.failures == []


async def test_orphan_task_is_quarantined_before_claim() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease(), contract_state=None)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = _delivery()

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "quarantined"
    assert repository.inspected == [delivery.message]
    assert events == ["quarantine", "ack"]
    assert queue.quarantined == [(delivery, "orphan_task")]
    assert repository.failures == []


async def test_inspect_failure_propagates_without_quarantining_or_acknowledging() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(
        events,
        claim=_lease(),
        inspect_error=RuntimeError("database unavailable"),
    )
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = _delivery()

    with pytest.raises(RuntimeError, match="database unavailable"):
        await runtime.handle_delivery(delivery)

    assert repository.inspected == [delivery.message]
    assert events == []
    assert queue.acknowledgements == []


async def test_normalized_legacy_delivery_passes_durable_contract_inspection() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease())
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    legacy = VideoTaskMessage.from_fields(_LEGACY_VIDEO_FIELDS)
    delivery = VideoTaskDelivery(message=legacy, receipt="11-0")

    outcome = await runtime.handle_delivery(delivery)

    assert legacy.legacy_wire
    assert outcome == "completed"
    assert repository.inspected == [legacy]


async def test_duplicate_or_fenced_attempt_is_acked_without_reprocessing_video() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=None)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )
    delivery = _delivery()

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "duplicate"
    assert events == ["claim", "ack"]
    assert queue.acknowledgements == [delivery]


async def test_active_delivery_reclaimed_by_idle_time_is_not_acked_early() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=None, ack_unclaimed=False)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )

    outcome = await runtime.handle_delivery(_delivery())

    assert outcome == "deferred"
    assert events == ["claim"]
    assert not queue.acknowledgements


async def test_stale_completion_is_fenced_and_only_acks_duplicate_delivery() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease(), complete_result=False)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_SuccessfulProcessor(),
        worker_id="worker-a",
    )

    outcome = await runtime.handle_delivery(_delivery())

    assert outcome == "duplicate"
    assert [event for event in events if event != "heartbeat"] == [
        "claim",
        "progress",
        "commit",
        "ack",
    ]
    assert not repository.retries
    assert not queue.dlq


async def test_failure_is_durably_scheduled_before_retry_ack_with_backoff() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease())
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_FailingProcessor(),
        worker_id="worker-a",
        retry_policy=RetryPolicy(max_attempts=2, retry_delays_seconds=(5.0,)),
        clock=lambda: 100.0,
    )
    delivery = _delivery()

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "retry_scheduled"
    assert repository.retries == [(_lease(), "ffmpeg progress timeout", 105.0)]
    assert [event for event in events if event != "heartbeat"] == [
        "claim",
        "schedule_retry",
        "retry",
    ]
    assert queue.retried == [delivery]
    assert not queue.acknowledgements


async def test_max_attempt_failure_enters_dlq_only_after_database_failure_commit() -> None:
    events: list[str] = []
    queue = _FakeQueue(events)
    repository = _FakeRepository(events, claim=_lease(attempt=2))
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=_FailingProcessor(),
        worker_id="worker-a",
        retry_policy=RetryPolicy(max_attempts=2, retry_delays_seconds=(5.0,)),
        clock=lambda: 100.0,
    )
    delivery = _delivery(attempt=1)

    outcome = await runtime.handle_delivery(delivery)

    assert outcome == "dlq"
    assert repository.failures == [(_lease(attempt=2), "ffmpeg progress timeout")]
    assert [event for event in events if event != "heartbeat"] == [
        "claim",
        "fail",
        "dlq",
        "mark_dlq",
        "ack",
    ]
    assert queue.dlq == [(delivery, "ffmpeg progress timeout")]
    assert queue.acknowledgements == [delivery]


@dataclass
class _FakeRecoveryRepository:
    events: list[str]
    messages: list[VideoTaskMessage]

    async def recover_timed_out(self, *, now: float) -> None:
        self.events.append(f"recover:{now}")

    async def dispatchable_messages(self, *, now: float) -> list[VideoTaskMessage]:
        self.events.append(f"due:{now}")
        return self.messages

    async def mark_published(self, message: VideoTaskMessage) -> None:
        self.events.append(f"mark:{message.task_id}")

    async def due_failed_dlq(
        self, *, now: float
    ) -> list[tuple[VideoTaskMessage, str]]:
        self.events.append(f"dlq_due:{now}")
        return []

    async def mark_dlq_published(self, _message: VideoTaskMessage) -> None:
        raise AssertionError("no DLQ item should be published")


@dataclass
class _FakePublishQueue:
    events: list[str]

    async def publish(self, message: VideoTaskMessage) -> str:
        self.events.append(f"publish:{message.task_id}")
        return "22-0"

    async def route_dlq(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no DLQ item should be published")


async def test_timeout_recovery_and_due_retry_redispatch_are_database_led() -> None:
    events: list[str] = []
    message = _message(attempt=1)
    scheduler = VideoTaskRecoveryScheduler(
        queue=_FakePublishQueue(events),
        repository=_FakeRecoveryRepository(events, [message]),
        clock=lambda: 123.0,
    )

    published = await scheduler.run_once()

    assert published == 1
    assert events == [
        "recover:123.0",
        "due:123.0",
        f"publish:{message.task_id}",
        f"mark:{message.task_id}",
        "dlq_due:123.0",
    ]
