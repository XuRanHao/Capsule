"""Create durable processing tasks from externally registered cloud sources."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from capsule.config import Settings, get_settings
from capsule.db.models import ProcessingJob, SourceFile
from capsule.db.processing_task_persistence import cpu_contract_for_extension
from capsule.db.session import Database
from capsule.db.video_tasks import VideoProcessingTask
from capsule.enums import JobStatus, PipelineStage, ProcessingStatus
from capsule.pipeline.video_task_runtime import ProcessingTaskKind, ResourceClass


class CloudSourceTaskDispatcher:
    """Claim externally uploaded S3 sources and persist one durable task per source."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database or Database(self._settings)
        self._owns_database = database is None

    async def run_once(self, *, limit: int = 100) -> int:
        """Convert at most ``limit`` pending cloud-source rows into queued tasks."""
        if limit < 1:
            raise ValueError("cloud source dispatch limit must be positive")
        created = 0
        async with self._database.session() as session, session.begin():
            rows = list(
                (
                    await session.scalars(
                        select(SourceFile)
                        .where(
                            SourceFile.processing_status == ProcessingStatus.PENDING.value,
                            SourceFile.storage_uri.like("s3://%"),
                        )
                        .order_by(SourceFile.created_at, SourceFile.source_file_id)
                        .limit(limit)
                        .with_for_update(skip_locked=True)
                    )
                ).all()
            )
            for source in rows:
                task_kind, resource_class, route_key = _route_for_source(source)
                existing = await session.scalar(
                    select(VideoProcessingTask.task_id).where(
                        VideoProcessingTask.source_file_id == source.source_file_id,
                        VideoProcessingTask.source_generation == source.processing_generation,
                        VideoProcessingTask.result_version == 1,
                    )
                )
                if existing is not None:
                    source.processing_status = ProcessingStatus.PROCESSING.value
                    continue
                job = ProcessingJob(
                    workspace_id=source.workspace_id,
                    job_type="cloud_source",
                    status=JobStatus.RUNNING.value,
                    current_stage=PipelineStage.PARSING.value,
                    input_path=source.storage_uri,
                    total_count=1,
                    started_at=datetime.now(UTC),
                )
                session.add(job)
                await session.flush()
                task = VideoProcessingTask(
                    parent_job_id=job.job_id,
                    source_file_id=source.source_file_id,
                    source_generation=source.processing_generation,
                    result_version=1,
                    task_kind=task_kind,
                    resource_class=resource_class,
                    route_key=route_key,
                    processor_version=1,
                    status="queued",
                    stage="queued",
                    attempt=0,
                    progress={},
                    input_payload={
                        "source_sha256": source.sha256,
                        "source_uri": source.storage_uri,
                        "processing_fingerprint": source.processing_fingerprint,
                    },
                )
                session.add(task)
                source.processing_status = ProcessingStatus.PROCESSING.value
                source.error_message = None
                created += 1
        return created

    async def run_forever(self, *, poll_seconds: float | None = None) -> None:
        delay = poll_seconds or self._settings.video_task_scheduler_poll_seconds
        if delay <= 0:
            raise ValueError("cloud source dispatch poll interval must be positive")
        while True:
            await self.run_once()
            await asyncio.sleep(delay)

    async def close(self) -> None:
        if self._owns_database:
            await self._database.dispose()


def _route_for_source(source: SourceFile) -> tuple[str, str, str]:
    extension = Path(source.relative_path).suffix.lower()
    if extension in {".mp4", ".mov"}:
        return (
            ProcessingTaskKind.VIDEO.value,
            ResourceClass.MPS_VIDEO.value,
            "mps_video",
        )
    if extension in {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}:
        return (
            ProcessingTaskKind.AUDIO.value,
            ResourceClass.MPS_VIDEO.value,
            "mps_video",
        )
    contract = cpu_contract_for_extension(extension)
    return contract.task_kind.value, contract.resource_class.value, contract.route_key
