"""Fake-only coverage for the operational durable-video wiring."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from capsule.config import Settings
from capsule.db.repositories import PreparedVideoTaskSubmission
from capsule.pipeline.video_task_runtime import VideoTaskMessage
from capsule.pipeline.video_task_service import (
    VideoTaskScheduler,
    VideoTaskSubmissionService,
    VideoTaskWorker,
)


@dataclass
class _FakeSourceRepository:
    events: list[str]

    async def create_video_task_submission(
        self, **_kwargs: Any
    ) -> PreparedVideoTaskSubmission:
        self.events.append("atomic_submit")
        return PreparedVideoTaskSubmission(
            job_id="job-1",
            source_file_id="source-1",
            generation=4,
            task_id="task-1",
            already_processed=False,
        )


@dataclass
class _FakeTaskRepository:
    events: list[str]

    async def mark_published(self, message: VideoTaskMessage) -> None:
        assert message.task_id == "task-1"
        self.events.append("mark_published")


@dataclass
class _FakeQueue:
    events: list[str]
    fail_publish: bool = False
    messages: list[VideoTaskMessage] = field(default_factory=list)

    async def start(self) -> None:
        self.events.append("queue_start")

    async def publish(self, message: VideoTaskMessage) -> str:
        self.events.append("xadd")
        self.messages.append(message)
        if self.fail_publish:
            raise ConnectionError("redis is unavailable")
        return "1-0"


async def test_submit_persists_task_before_starting_and_publishing_redis(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"not decoded by this test")
    events: list[str] = []
    tasks = _FakeTaskRepository(events)
    queue = _FakeQueue(events)
    service = VideoTaskSubmissionService(
        settings=Settings(video_source_roots=[tmp_path]),
        source_repository=_FakeSourceRepository(events),
        task_repository=tasks,
        queue=queue,
    )

    result = await service.submit(input_path=video, workspace_id="workspace-1")

    assert events == [
        "atomic_submit",
        "queue_start",
        "xadd",
        "mark_published",
    ]
    assert result.published and result.task_id == "task-1"
    assert queue.messages[0].source_uri == video.resolve().as_uri()


async def test_submit_redis_failure_keeps_created_task_queued(tmp_path: Path) -> None:
    video = tmp_path / "demo.mov"
    video.write_bytes(b"not decoded by this test")
    events: list[str] = []
    tasks = _FakeTaskRepository(events)
    queue = _FakeQueue(events, fail_publish=True)
    service = VideoTaskSubmissionService(
        settings=Settings(video_source_roots=[tmp_path]),
        source_repository=_FakeSourceRepository(events),
        task_repository=tasks,
        queue=queue,
    )

    with pytest.raises(ConnectionError, match="redis is unavailable"):
        await service.submit(input_path=video, workspace_id="workspace-1")

    assert events == ["atomic_submit", "queue_start", "xadd"]
    assert len(queue.messages) == 1


async def test_submit_does_not_touch_redis_when_atomic_database_submit_fails(
    tmp_path: Path,
) -> None:
    class FailingSource(_FakeSourceRepository):
        async def create_video_task_submission(
            self, **_kwargs: Any
        ) -> PreparedVideoTaskSubmission:
            self.events.append("atomic_submit")
            raise RuntimeError("database transaction rolled back")

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"not decoded by this test")
    events: list[str] = []
    queue = _FakeQueue(events)
    service = VideoTaskSubmissionService(
        settings=Settings(video_source_roots=[tmp_path]),
        source_repository=FailingSource(events),
        task_repository=_FakeTaskRepository(events),
        queue=queue,
    )

    with pytest.raises(RuntimeError, match="transaction rolled back"):
        await service.submit(input_path=video, workspace_id="workspace-1")

    assert events == ["atomic_submit"]
    assert queue.messages == []


async def test_submit_reused_source_closes_its_new_parent_job(tmp_path: Path) -> None:
    class ReusedSource(_FakeSourceRepository):
        async def create_video_task_submission(
            self, **_kwargs: Any
        ) -> PreparedVideoTaskSubmission:
            self.events.append("atomic_submit")
            return PreparedVideoTaskSubmission(
                job_id="job-1",
                source_file_id="source-1",
                generation=4,
                task_id=None,
                already_processed=True,
            )

    video = tmp_path / "demo.mp4"
    video.write_bytes(b"already processed")
    events: list[str] = []
    service = VideoTaskSubmissionService(
        settings=Settings(video_source_roots=[tmp_path]),
        source_repository=ReusedSource(events),
        task_repository=_FakeTaskRepository(events),
        queue=_FakeQueue(events),
    )

    result = await service.submit(input_path=video, workspace_id="workspace-1")

    assert result.already_processed and not result.published
    assert events == ["atomic_submit"]


async def test_submit_rejects_an_unconfigured_external_video_root(tmp_path: Path) -> None:
    video = tmp_path / "external.mp4"
    video.write_bytes(b"video")
    events: list[str] = []
    service = VideoTaskSubmissionService(
        settings=Settings(import_root=tmp_path / "imports"),
        source_repository=_FakeSourceRepository(events),
        task_repository=_FakeTaskRepository(events),
        queue=_FakeQueue(events),
    )

    with pytest.raises(ValueError, match="CAPSULE_VIDEO_SOURCE_ROOTS"):
        await service.submit(input_path=video, workspace_id="workspace-1")

    assert events == []


async def test_submit_accepts_a_configured_video_source_root(tmp_path: Path) -> None:
    video = tmp_path / "allowed.mp4"
    video.write_bytes(b"video")
    events: list[str] = []
    service = VideoTaskSubmissionService(
        settings=Settings(video_source_roots=[tmp_path]),
        source_repository=_FakeSourceRepository(events),
        task_repository=_FakeTaskRepository(events),
        queue=_FakeQueue(events),
    )

    result = await service.submit(input_path=video, workspace_id="workspace-1")

    assert result.published
    assert events[:2] == ["atomic_submit", "queue_start"]


class _FakeWorkerQueue:
    def __init__(self) -> None:
        self.started = 0

    async def start(self) -> None:
        self.started += 1

    async def receive(self) -> str:
        return "delivery-1"


class _FakeRuntime:
    def __init__(self) -> None:
        self.deliveries: list[str] = []

    async def handle_delivery(self, delivery: str) -> str:
        self.deliveries.append(delivery)
        return "completed"


async def test_worker_run_once_starts_queue_and_handles_one_delivery() -> None:
    queue = _FakeWorkerQueue()
    runtime = _FakeRuntime()
    worker = VideoTaskWorker(runtime=runtime, queue=queue)  # type: ignore[arg-type]

    assert await worker.run_once() == "completed"
    assert queue.started == 1
    assert runtime.deliveries == ["delivery-1"]


async def test_worker_run_forever_processes_two_deliveries_concurrently() -> None:
    class Queue:
        def __init__(self) -> None:
            self.started = 0
            self.next_delivery = 0

        async def start(self) -> None:
            self.started += 1

        async def receive(self) -> str:
            self.next_delivery += 1
            if self.next_delivery <= 2:
                return f"delivery-{self.next_delivery}"
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class Runtime:
        def __init__(self) -> None:
            self.deliveries: list[str] = []
            self.both_started = asyncio.Event()

        async def handle_delivery(self, delivery: str) -> str:
            self.deliveries.append(delivery)
            if len(self.deliveries) == 2:
                self.both_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    queue = Queue()
    runtime = Runtime()
    worker = VideoTaskWorker(
        runtime=runtime,  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        concurrency=2,
    )

    running = asyncio.create_task(worker.run_forever())
    await asyncio.wait_for(runtime.both_started.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert queue.started == 1
    assert runtime.deliveries == ["delivery-1", "delivery-2"]


class _FakeSchedulerQueue:
    def __init__(self) -> None:
        self.started = 0

    async def start(self) -> None:
        self.started += 1


class _FakeScheduler:
    async def run_once(self) -> int:
        return 3


async def test_scheduler_run_once_starts_transport_and_reports_published_count() -> None:
    queue = _FakeSchedulerQueue()
    scheduler = VideoTaskScheduler(
        scheduler=_FakeScheduler(),  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        poll_seconds=0.01,
    )

    assert await scheduler.run_once() == 3
    assert queue.started == 1
