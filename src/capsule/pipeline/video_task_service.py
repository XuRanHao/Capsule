"""Operational wiring for the durable whole-video task pipeline.

The PostgreSQL task record is deliberately written before Redis is touched.  A
Redis outage therefore leaves recoverable ``queued`` work for the scheduler;
Redis Streams remain a delivery mechanism, not the source of truth.
"""

from __future__ import annotations

import asyncio
import mimetypes
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capsule.config import Settings, get_settings
from capsule.db.repositories import AssetRepository, PreparedVideoTaskSubmission
from capsule.db.session import Database
from capsule.db.video_asset_committer import PostgresFencedVideoAssetCommitter
from capsule.db.video_tasks import PostgresVideoTaskRepository
from capsule.model_clients.mobileclip import ResidentMobileClipWorker
from capsule.parsers.discovery import sha256_file
from capsule.parsers.video import VideoParser, VideoSegmentationConfig
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.runner import _processing_fingerprint
from capsule.pipeline.video_media import VideoDerivedMediaWriter
from capsule.pipeline.video_task_processor import CapsuleVideoTaskProcessor
from capsule.pipeline.video_task_runtime import (
    RedisVideoTaskQueue,
    RetryPolicy,
    VideoTaskMessage,
    VideoTaskQueue,
    VideoTaskRecoveryScheduler,
    VideoTaskRuntime,
)
from capsule.schemas import DiscoveredFile
from capsule.storage.object_storage import ObjectStorage
from capsule.video_sources import validate_video_source_root


class _SourceRepository(Protocol):
    async def create_video_task_submission(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        result_version: int = 1,
    ) -> PreparedVideoTaskSubmission: ...


class _TaskRepository(Protocol):
    async def mark_published(self, message: VideoTaskMessage) -> None: ...


@dataclass(frozen=True, slots=True)
class VideoTaskSubmission:
    """The durable task identity and whether existing output was reused."""

    job_id: str
    source_file_id: str
    generation: int
    task_id: str | None
    published: bool
    already_processed: bool = False


class VideoTaskSubmissionService:
    """Durably submit one local MP4/MOV for independent video processing."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
        source_repository: _SourceRepository | None = None,
        task_repository: _TaskRepository | None = None,
        queue: VideoTaskQueue | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._source_repository = source_repository
        self._task_repository = task_repository
        self._queue = queue

    async def submit(self, *, input_path: Path, workspace_id: str) -> VideoTaskSubmission:
        source_file = await asyncio.to_thread(_video_source, input_path)
        validate_video_source_root(
            Path(source_file.path),
            import_root=self._settings.import_root,
            video_source_roots=self._settings.video_source_roots,
        )
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        source_repository = self._source_repository or AssetRepository(database)
        task_repository = self._task_repository or PostgresVideoTaskRepository(
            database.session_factory,
            lease_seconds=self._settings.video_task_lease_seconds,
            progress_timeout_seconds=self._settings.video_task_progress_timeout_seconds,
            hard_timeout_seconds=self._settings.video_task_hard_timeout_seconds,
            redispatch_seconds=self._settings.video_task_redispatch_seconds,
            max_attempts=self._settings.video_task_max_attempts,
        )
        queue = self._queue or _new_queue(self._settings, consumer=_submission_consumer())
        owns_queue = self._queue is None
        try:
            digest = await asyncio.to_thread(sha256_file, Path(source_file.path))
            # The parent job, source generation, and queued task commit together.
            # If Redis is unavailable after this point, video-scheduler republishes
            # the durable queued row; a database failure leaves none of the three.
            submitted = await source_repository.create_video_task_submission(
                workspace_id=workspace_id,
                input_path=Path(source_file.path),
                source_file=source_file,
                sha256=digest,
                mime_type=_mime_type(source_file),
                processing_fingerprint=_processing_fingerprint(source_file, self._settings),
            )
            if submitted.already_processed:
                return VideoTaskSubmission(
                    job_id=submitted.job_id,
                    source_file_id=submitted.source_file_id,
                    generation=submitted.generation,
                    task_id=None,
                    published=False,
                    already_processed=True,
                )
            if submitted.task_id is None:  # pragma: no cover - repository invariant
                raise RuntimeError("queued video submission did not create a task id")
            message = VideoTaskMessage(
                job_id=submitted.job_id,
                workspace_id=workspace_id,
                source_file_id=submitted.source_file_id,
                generation=submitted.generation,
                task_id=submitted.task_id,
                source_uri=Path(source_file.path).as_uri(),
            )
            await _start_queue(queue)
            await queue.publish(message)
            await task_repository.mark_published(message)
            return VideoTaskSubmission(
                job_id=submitted.job_id,
                source_file_id=submitted.source_file_id,
                generation=submitted.generation,
                task_id=message.task_id,
                published=True,
            )
        finally:
            if owns_queue:
                await _close_queue(queue)
            if owns_database:
                await database.dispose()


class VideoTaskWorker:
    """Own the Redis consumer and bounded MPS/video slots for one worker process."""

    def __init__(
        self,
        *,
        runtime: VideoTaskRuntime,
        queue: VideoTaskQueue,
        close: Callable[[], Awaitable[None]] | None = None,
        concurrency: int = 1,
    ) -> None:
        if concurrency < 1:
            raise ValueError("video worker concurrency must be positive")
        self._runtime = runtime
        self._queue = queue
        self._close = close
        self._concurrency = concurrency
        self._started = False

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Settings | None = None,
        worker_id: str | None = None,
    ) -> VideoTaskWorker:
        runtime_settings = settings or get_settings()
        database = Database(runtime_settings)
        queue = _new_queue(runtime_settings, consumer=worker_id or _worker_identity())
        parser = VideoParser(
            concurrency=runtime_settings.ffmpeg_concurrency,
            config=_video_config(runtime_settings),
            embedder=ResidentMobileClipWorker(
                model_path=Path(runtime_settings.mobileclip_model_path),
                batch_size=runtime_settings.mobileclip_batch_size,
            ),
        )
        writer: VideoDerivedMediaWriter | None = None
        if runtime_settings.video_output_mode == "materialized":
            writer = VideoDerivedMediaWriter(
                ObjectStorage(runtime_settings),
                concurrency=runtime_settings.ffmpeg_concurrency,
                upload_concurrency=runtime_settings.video_upload_concurrency,
                spool_root=runtime_settings.video_spool_root,
                spool_max_items=runtime_settings.video_spool_max_items,
                spool_max_bytes=runtime_settings.video_spool_max_bytes,
                queue_backend=runtime_settings.video_upload_queue_backend,
                redis_url=runtime_settings.redis_url,
                redis_stream=runtime_settings.video_upload_stream,
                redis_group=runtime_settings.video_upload_group,
                redis_claim_idle_ms=runtime_settings.video_upload_claim_idle_ms,
                max_upload_attempts=runtime_settings.video_upload_max_attempts,
                retry_base_seconds=runtime_settings.video_upload_retry_base_seconds,
            )
        processor = CapsuleVideoTaskProcessor(
            parser=parser,
            asset_factory=AssetFactory(),
            media_writer=writer,
            committer=PostgresFencedVideoAssetCommitter(database),
            output_mode=runtime_settings.video_output_mode,
        )
        runtime = VideoTaskRuntime(
            queue=queue,
            repository=PostgresVideoTaskRepository(
                database.session_factory,
                lease_seconds=runtime_settings.video_task_lease_seconds,
                progress_timeout_seconds=(
                    runtime_settings.video_task_progress_timeout_seconds
                ),
                hard_timeout_seconds=runtime_settings.video_task_hard_timeout_seconds,
                redispatch_seconds=runtime_settings.video_task_redispatch_seconds,
                max_attempts=runtime_settings.video_task_max_attempts,
            ),
            processor=processor,
            worker_id=worker_id or _worker_identity(),
            retry_policy=RetryPolicy(
                max_attempts=runtime_settings.video_task_max_attempts,
                retry_delays_seconds=tuple(runtime_settings.video_task_retry_delays_seconds),
            ),
            heartbeat_seconds=runtime_settings.video_task_heartbeat_seconds,
            expected_route_key="mps_video",
        )

        async def close() -> None:
            if writer is not None:
                await writer.close()
            await _close_queue(queue)
            await database.dispose()

        return cls(
            runtime=runtime,
            queue=queue,
            close=close,
            concurrency=runtime_settings.ffmpeg_concurrency,
        )

    async def run_once(self) -> str:
        await self.start()
        return await self._runtime.handle_delivery(await self._queue.receive())

    async def run_forever(self) -> None:
        await self.start()
        slots = [asyncio.create_task(self._run_slot()) for _ in range(self._concurrency)]
        try:
            await asyncio.gather(*slots)
        finally:
            for slot in slots:
                slot.cancel()
            await asyncio.gather(*slots, return_exceptions=True)
            await self.close()

    async def _run_slot(self) -> None:
        while True:
            await self._runtime.handle_delivery(await self._queue.receive())

    async def close(self) -> None:
        if self._close is not None:
            close, self._close = self._close, None
            await close()

    async def start(self) -> None:
        """Open the queue before the runtime receives its first delivery."""
        if not self._started:
            await _start_queue(self._queue)
            self._started = True


class VideoTaskScheduler:
    """Run durable timeout recovery and retry publication outside worker processes."""

    def __init__(
        self,
        *,
        scheduler: VideoTaskRecoveryScheduler,
        queue: VideoTaskQueue,
        poll_seconds: float,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._queue = queue
        self._poll_seconds = poll_seconds
        self._close = close
        self._started = False

    @classmethod
    def from_settings(cls, *, settings: Settings | None = None) -> VideoTaskScheduler:
        runtime_settings = settings or get_settings()
        database = Database(runtime_settings)
        queue = _new_queue(runtime_settings, consumer=f"scheduler-{_worker_identity()}")

        async def close() -> None:
            await _close_queue(queue)
            await database.dispose()

        return cls(
            scheduler=VideoTaskRecoveryScheduler(
                queue=queue,
                repository=PostgresVideoTaskRepository(
                    database.session_factory,
                    lease_seconds=runtime_settings.video_task_lease_seconds,
                    progress_timeout_seconds=(
                        runtime_settings.video_task_progress_timeout_seconds
                    ),
                    hard_timeout_seconds=runtime_settings.video_task_hard_timeout_seconds,
                    redispatch_seconds=runtime_settings.video_task_redispatch_seconds,
                    max_attempts=runtime_settings.video_task_max_attempts,
                ),
            ),
            queue=queue,
            poll_seconds=runtime_settings.video_task_scheduler_poll_seconds,
            close=close,
        )

    async def run_once(self) -> int:
        await self.start()
        return await self._scheduler.run_once()

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await self._scheduler.run_once()
                await asyncio.sleep(self._poll_seconds)
        finally:
            await self.close()

    async def close(self) -> None:
        if self._close is not None:
            close, self._close = self._close, None
            await close()

    async def start(self) -> None:
        """Open the queue before the first durable recovery pass."""
        if not self._started:
            await _start_queue(self._queue)
            self._started = True


def _new_queue(settings: Settings, *, consumer: str) -> RedisVideoTaskQueue:
    return RedisVideoTaskQueue(
        redis_url=settings.redis_url,
        stream=settings.video_task_stream,
        group=settings.video_task_group,
        consumer=consumer,
        dlq_stream=settings.video_task_dlq_stream,
        claim_idle_ms=settings.video_task_claim_idle_ms,
    )


def _video_config(settings: Settings) -> VideoSegmentationConfig:
    return VideoSegmentationConfig(
        output_mode=settings.video_output_mode,
        sample_interval_seconds=settings.video_sample_interval_seconds,
        min_segment_seconds=settings.video_min_segment_seconds,
        distance_quantile=settings.video_distance_quantile,
        min_distance_threshold=settings.video_min_distance_threshold,
        max_distance_threshold=settings.video_max_distance_threshold,
        similarity_relaxation=settings.video_similarity_relaxation,
        max_merge_cost=settings.video_max_merge_cost,
        base_target_seconds=settings.video_base_target_seconds,
        max_target_seconds=settings.video_max_target_seconds,
        target_log2_weight=settings.video_target_log2_weight,
        hard_max_duration_factor=settings.video_hard_max_duration_factor,
        activity_sample_fps=settings.video_activity_sample_fps,
        activity_envelope_seconds=settings.video_activity_envelope_seconds,
        activity_shift_side_seconds=settings.video_activity_shift_side_seconds,
        keyframe_size=settings.video_keyframe_size,
        keyframe_jpeg_quality=settings.video_keyframe_jpeg_quality,
        max_representative_frames=settings.video_max_representative_frames,
    )


def _video_source(input_path: Path) -> DiscoveredFile:
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"video task source does not exist: {path}")
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("submit-video-task accepts a single .mp4 or .mov file")
    return DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )


def _mime_type(source_file: DiscoveredFile) -> str:
    guessed, _ = mimetypes.guess_type(source_file.path)
    return guessed or "application/octet-stream"


def _worker_identity() -> str:
    return f"{socket.gethostname()}-{__import__('os').getpid()}"


def _submission_consumer() -> str:
    return f"submit-{_worker_identity()}"


async def _start_queue(queue: VideoTaskQueue) -> None:
    start = getattr(queue, "start", None)
    if start is not None:
        await start()


async def _close_queue(queue: VideoTaskQueue) -> None:
    close = getattr(queue, "close", None)
    if close is not None:
        await close()
