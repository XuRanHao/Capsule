from pathlib import Path

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile, Workspace
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.db.video_asset_committer import PostgresFencedVideoAssetCommitter
from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.enums import AssetType, ProcessingStatus
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_runtime import VideoTaskMessage, VideoTaskResult
from capsule.schemas import AssetDraft, DiscoveredFile


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fenced_segment_commit_publishes_only_complete_generation_once(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_video_task_committer"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
                task_table = await session.scalar(
                    text("select to_regclass('public.video_processing_tasks')")
                )
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")
        if task_table is None:
            pytest.skip("PostgreSQL integration database has not run video task migration")

        path = tmp_path / "demo.mp4"
        path.write_bytes(b"video-bytes")
        discovered = DiscoveredFile(
            path=str(path),
            relative_path=path.name,
            extension=".mp4",
            size_bytes=path.stat().st_size,
        )
        assets = AssetRepository(database)
        await assets.create_workspace(name="Video task committer", workspace_id=workspace_id)
        submitted = await assets.create_video_task_submission(
            workspace_id=workspace_id,
            input_path=tmp_path,
            source_file=discovered,
            sha256=sha256_file(path),
            mime_type="video/mp4",
            processing_fingerprint="v" * 64,
        )
        assert not submitted.already_processed
        assert submitted.task_id is not None
        message = VideoTaskMessage(
            job_id=submitted.job_id,
            workspace_id=workspace_id,
            source_file_id=submitted.source_file_id,
            generation=submitted.generation,
            task_id=submitted.task_id,
            result_version=1,
            source_uri=path.as_uri(),
        )
        tasks = PostgresVideoTaskRepository(database.session_factory)
        async with database.session() as session:
            submitted_job = await session.get(ProcessingJob, submitted.job_id)
            submitted_source = await session.get(SourceFile, submitted.source_file_id)
            submitted_task = await session.get(VideoProcessingTask, submitted.task_id)
        assert submitted_job is not None and submitted_job.status == "running"
        assert submitted_source is not None
        assert submitted_source.processing_status == ProcessingStatus.PROCESSING.value
        assert submitted_source.processing_generation == submitted.generation
        assert submitted_task is not None
        assert submitted_task.status == "queued"
        assert submitted_task.parent_job_id == submitted.job_id
        lease = await tasks.claim_attempt(message, worker_id="worker-1", receipt="1-0")
        assert lease is not None

        factory = AssetFactory()
        task_assets = factory.build_many(
            workspace_id=workspace_id,
            source_file_id=submitted.source_file_id,
            source_sha256=sha256_file(path),
            source_file=discovered,
            generation=submitted.generation,
            drafts=[
                AssetDraft(
                    asset_type=AssetType.VIDEO_SEGMENT,
                    file_name=path.name,
                    source_locator={"start_ms": 0, "end_ms": 1000},
                ),
                AssetDraft(
                    asset_type=AssetType.VIDEO_SEGMENT,
                    file_name=path.name,
                    source_locator={"start_ms": 1000, "end_ms": 2000},
                ),
            ],
        )
        committer = PostgresFencedVideoAssetCommitter(database)
        assert await committer.validate_lease_source(message, lease)
        await committer.commit_segment(message, lease, task_assets[0], expected_asset_count=2)

        async with database.session() as session:
            source = await session.get(SourceFile, submitted.source_file_id)
            job = await session.get(ProcessingJob, submitted.job_id)
        assert source is not None and source.processing_status == ProcessingStatus.PROCESSING.value
        assert job is not None and job.completed_count == 0

        committed_id = await committer.commit_segment(
            message, lease, task_assets[1], expected_asset_count=2
        )
        replayed_id = await committer.commit_segment(
            message, lease, task_assets[1], expected_asset_count=2
        )
        assert replayed_id == committed_id
        assert await tasks.complete(
            lease,
            VideoTaskResult(result_ref="video-task://demo", metadata={"asset_count": 2}),
        )
        async with database.session() as session:
            source = await session.get(SourceFile, submitted.source_file_id)
            job = await session.get(ProcessingJob, submitted.job_id)
            task = await session.get(VideoProcessingTask, message.task_id)
            stored = list(
                await session.scalars(
                    select(Asset).where(Asset.source_file_id == submitted.source_file_id)
                )
            )
        assert source is not None and source.processing_status == ProcessingStatus.COMPLETED.value
        assert job is not None and job.completed_count == 1
        assert job.status == "completed"
        assert task is not None
        assert task.status == "completed"
        assert task.parent_accounted_at is not None
        assert len(stored) == 2
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()
