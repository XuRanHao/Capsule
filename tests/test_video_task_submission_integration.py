import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, event, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import ProcessingJob, SourceFile, Workspace
from capsule.db.repositories import AssetRepository, ProcessingTaskSourceInput
from capsule.db.session import Database
from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.pipeline.video_task_service import VideoTaskSubmissionService
from capsule.schemas import DiscoveredFile


async def _require_video_task_database(database: Database) -> None:
    try:
        async with database.session() as session:
            await session.execute(text("select 1"))
            task_table = await session.scalar(
                text("select to_regclass('public.processing_tasks')")
            )
            input_payload_column = await session.scalar(
                text(
                    "select exists ("
                    "select 1 from information_schema.columns "
                    "where table_schema = 'public' "
                    "and table_name = 'processing_tasks' "
                    "and column_name = 'input_payload'"
                    ")"
                )
            )
    except SQLAlchemyError:
        pytest.skip("PostgreSQL integration database is unavailable")
    if task_table is None or not input_payload_column:
        pytest.skip("PostgreSQL integration database has not run video task migration")


async def _delete_workspace(database: Database, workspace_id: str) -> None:
    async with database.session() as session, session.begin():
        await session.execute(delete(Workspace).where(Workspace.workspace_id == workspace_id))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_atomic_submission_rolls_back_job_and_source_when_task_insert_fails(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    database_ready = False
    workspace_id = "workspace_video_task_submit_rollback"
    video = tmp_path / "rollback.mp4"
    video.write_bytes(b"video-task-rollback")
    settings = get_settings().model_copy(update={"video_source_roots": [tmp_path]})

    def fail_task_insert(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected video task insert failure")

    try:
        await _require_video_task_database(database)
        database_ready = True
        await _delete_workspace(database, workspace_id)
        event.listen(VideoProcessingTask, "before_insert", fail_task_insert)
        try:
            with pytest.raises(RuntimeError, match="injected video task insert failure"):
                await VideoTaskSubmissionService(
                    settings=settings,
                    database=database,
                ).submit(
                    input_path=video,
                    workspace_id=workspace_id,
                )
        finally:
            event.remove(VideoProcessingTask, "before_insert", fail_task_insert)

        async with database.session() as session:
            workspace = await session.get(Workspace, workspace_id)
            jobs = list(
                await session.scalars(
                    select(ProcessingJob).where(ProcessingJob.workspace_id == workspace_id)
                )
            )
            sources = list(
                await session.scalars(
                    select(SourceFile).where(SourceFile.workspace_id == workspace_id)
                )
            )
        assert workspace is None
        assert jobs == []
        assert sources == []
    finally:
        if database_ready:
            await _delete_workspace(database, workspace_id)
        await database.dispose()


class _FailingPublishQueue:
    async def start(self) -> None:
        return None

    async def publish(self, _message: object) -> str:
        raise ConnectionError("injected Redis XADD failure")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_browser_batch_persists_canonical_uri_and_rebuilds_after_dispatch_loss(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    database_ready = False
    workspace_id = "workspace_browser_batch_dispatch_recovery"
    image = tmp_path / "nested" / "image.png"
    image.parent.mkdir()
    image.write_bytes(b"browser batch image")
    source_input = ProcessingTaskSourceInput(
        source_file=DiscoveredFile(
            path=str(image),
            relative_path="nested/image.png",
            extension=".png",
            size_bytes=image.stat().st_size,
        ),
        # A repository caller must not be able to persist an inconsistent
        # transport URI: SourceFile is the durable canonical source.
        source_uri="file:///untrusted/browser/path.png",
        sha256="d" * 64,
        mime_type="image/png",
        processing_fingerprint="browser-batch-v1",
    )
    try:
        await _require_video_task_database(database)
        database_ready = True
        await _delete_workspace(database, workspace_id)
        repository = AssetRepository(database)
        job_id = await repository.create_pending_import_job(
            workspace_id=workspace_id,
            import_root=tmp_path,
        )

        prepared = await repository.prepare_import_task_batch(
            job_id=job_id,
            workspace_id=workspace_id,
            items=[source_input],
        )
        replayed = await repository.prepare_import_task_batch(
            job_id=job_id,
            workspace_id=workspace_id,
            items=[source_input],
        )

        assert len(prepared) == 1
        assert replayed == prepared
        assert prepared[0].source_uri == image.resolve().as_uri()

        async with database.session() as session:
            job = await session.get(ProcessingJob, job_id)
            source = await session.get(SourceFile, prepared[0].source_file_id)
            task = await session.get(VideoProcessingTask, prepared[0].task_id)
        assert job is not None
        assert job.status == "running"
        assert job.total_count == 1
        assert job.dispatch_completed_at is not None
        assert source is not None and source.storage_uri == image.resolve().as_uri()
        assert task is not None and task.status == "queued"
        assert task.input_payload["source_uri"] == source.storage_uri
        assert task.input_payload["source_sha256"] == source.sha256

        task_repository = PostgresVideoTaskRepository(
            database.session_factory,
            task_kind="image",
            resource_class="cpu",
            route_key="cpu_image",
        )
        rebuilt = await task_repository.dispatchable_messages(now=time.time())
        message = next(item for item in rebuilt if item.task_id == prepared[0].task_id)
        assert message.source_uri == source.storage_uri
        assert await task_repository.inspect_message_contract(message)
    finally:
        if database_ready:
            await _delete_workspace(database, workspace_id)
        await database.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_redis_failure_leaves_queued_task_rebuildable_from_postgres(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    database_ready = False
    workspace_id = "workspace_video_task_submit_redis_failure"
    video = tmp_path / "redis-failure.mp4"
    video.write_bytes(b"video-task-redis-failure")
    settings = get_settings().model_copy(update={"video_source_roots": [tmp_path]})
    repository = PostgresVideoTaskRepository(database.session_factory)

    try:
        await _require_video_task_database(database)
        database_ready = True
        await _delete_workspace(database, workspace_id)
        service = VideoTaskSubmissionService(
            settings=settings,
            database=database,
            task_repository=repository,
            queue=_FailingPublishQueue(),  # type: ignore[arg-type]
        )
        with pytest.raises(ConnectionError, match="injected Redis XADD failure"):
            await service.submit(input_path=video, workspace_id=workspace_id)

        async with database.session() as session:
            job = await session.scalar(
                select(ProcessingJob).where(ProcessingJob.workspace_id == workspace_id)
            )
            source = await session.scalar(
                select(SourceFile).where(SourceFile.workspace_id == workspace_id)
            )
            task = await session.scalar(
                select(VideoProcessingTask)
                .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
                .where(ProcessingJob.workspace_id == workspace_id)
            )
        assert job is not None and job.status == "running"
        assert source is not None and source.processing_status == "processing"
        assert task is not None and task.status == "queued"
        assert task.last_published_at is None

        dispatchable = await repository.dispatchable_messages(now=time.time())
        rebuilt = next(message for message in dispatchable if message.task_id == task.task_id)
        assert rebuilt.job_id == job.job_id
        assert rebuilt.source_file_id == source.source_file_id
        assert rebuilt.source_uri == video.resolve().as_uri()
        assert await repository.inspect_message_contract(rebuilt)
    finally:
        if database_ready:
            await _delete_workspace(database, workspace_id)
        await database.dispose()
