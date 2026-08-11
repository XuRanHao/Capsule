"""Reliability contract for the durable whole-video task runtime.

These tests deliberately use fakes: correctness of the state-machine must not
depend on a locally running Redis or PostgreSQL service.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from capsule.pipeline.video_task_runtime import (
    RedisVideoTaskQueue,
    RetryPolicy,
    VideoTaskDelivery,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskRecoveryScheduler,
    VideoTaskResult,
    VideoTaskRuntime,
)


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
        raise AssertionError("pending entries must be claimed before reading new work")

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
    retries: list[tuple[VideoTaskLease, str, float]] = field(default_factory=list)
    failures: list[tuple[VideoTaskLease, str]] = field(default_factory=list)

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
