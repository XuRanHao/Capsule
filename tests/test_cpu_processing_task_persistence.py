"""Durable CPU image/text task contracts and atomic submission coverage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile, Workspace
from capsule.db.processing_task_persistence import (
    CPU_IMAGE_ROUTE,
    CPU_TEXT_ROUTE,
    PostgresCpuProcessingTaskRepository,
    PostgresFencedAssetBatchCommitter,
    PostgresFencedProcessingAssetCommitter,
    cpu_contract_for_extension,
)
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.db.video_asset_committer import _owned_live_lease_clauses
from capsule.db.video_tasks import VideoProcessingTask
from capsule.enums import AssetType, ProcessingStatus
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    ProcessingTaskKind,
    ResourceClass,
    VideoTaskLease,
    VideoTaskMessage,
)
from capsule.schemas import AssetDraft, DiscoveredFile


@pytest.mark.parametrize(
    ("extension", "task_kind", "route_key"),
    [
        (".png", ProcessingTaskKind.IMAGE, CPU_IMAGE_ROUTE),
        (".JPEG", ProcessingTaskKind.IMAGE, CPU_IMAGE_ROUTE),
        (".txt", ProcessingTaskKind.TEXT, CPU_TEXT_ROUTE),
        (".PDF", ProcessingTaskKind.TEXT, CPU_TEXT_ROUTE),
    ],
)
def test_cpu_contract_maps_only_supported_image_and_text_extensions(
    extension: str, task_kind: ProcessingTaskKind, route_key: str
) -> None:
    contract = cpu_contract_for_extension(extension)

    assert contract.task_kind is task_kind
    assert contract.resource_class is ResourceClass.CPU
    assert contract.route_key == route_key


def test_cpu_contract_rejects_video_and_unknown_extensions() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        cpu_contract_for_extension(".mp4")
    with pytest.raises(ValueError, match="unsupported"):
        cpu_contract_for_extension(".exe")


def test_cpu_repository_is_bound_to_one_shared_route_and_kind() -> None:
    repository = PostgresCpuProcessingTaskRepository(
        cast(async_sessionmaker[AsyncSession], None),
        task_kind=ProcessingTaskKind.IMAGE,
        processor_version=3,
    )
    message = VideoTaskMessage(
        task_id="image-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
        task_kind=ProcessingTaskKind.IMAGE,
        resource_class=ResourceClass.CPU,
        route_key=CPU_IMAGE_ROUTE,
        processor_version=3,
    )

    assert repository._message_matches_identity(message)
    assert not repository._message_matches_identity(replace(message, route_key="another-cpu-route"))


def test_cpu_asset_committer_inherits_complete_identity_fencing() -> None:
    lease = VideoTaskLease(
        task_id="image-task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="cpu-worker-1",
        result_version=1,
        task_kind=ProcessingTaskKind.IMAGE,
        resource_class=ResourceClass.CPU,
        route_key=CPU_IMAGE_ROUTE,
        processor_version=3,
        lease_token="claim-token",
    )
    clauses = " AND ".join(str(clause) for clause in _owned_live_lease_clauses(lease))

    assert issubclass(PostgresFencedProcessingAssetCommitter, object)
    assert all(
        column in clauses
        for column in (
            "processing_tasks.task_kind",
            "processing_tasks.resource_class",
            "processing_tasks.route_key",
            "processing_tasks.processor_version",
            "processing_tasks.lease_token",
        )
    )


@pytest.mark.asyncio
async def test_batch_committer_rejects_a_message_that_does_not_match_its_lease() -> None:
    message = VideoTaskMessage(
        task_id="image-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
        task_kind=ProcessingTaskKind.IMAGE,
        resource_class=ResourceClass.CPU,
        route_key=CPU_IMAGE_ROUTE,
    )
    lease = VideoTaskLease(
        task_id="image-task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="cpu-worker-1",
        result_version=1,
        task_kind=ProcessingTaskKind.IMAGE,
        resource_class=ResourceClass.CPU,
        route_key=CPU_IMAGE_ROUTE,
        lease_token="claim-token",
    )
    committer = PostgresFencedAssetBatchCommitter(None)

    assert not await committer.validate_lease_source(
        replace(message, route_key="wrong-route"), lease
    )
    with pytest.raises(LeaseLostError, match="does not match"):
        await committer.commit_assets(
            replace(message, route_key="wrong-route"),
            lease,
            [],
            source_sha256="a" * 64,
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extension", "expected_kind", "route_key"),
    [(".png", "image", CPU_IMAGE_ROUTE), (".txt", "text", CPU_TEXT_ROUTE)],
)
async def test_cpu_submission_atomically_creates_job_source_and_queued_task(
    tmp_path: Path,
    extension: str,
    expected_kind: str,
    route_key: str,
) -> None:
    database = Database(get_settings())
    workspace_id = f"workspace_cpu_task_{expected_kind}"
    path = tmp_path / f"sample{extension}"
    path.write_bytes(b"cpu task source")
    discovered = DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=extension,
        size_bytes=path.stat().st_size,
    )
    try:
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
            pytest.skip("PostgreSQL integration database has not run task migration")

        repository = AssetRepository(database)
        await repository.create_workspace(name=workspace_id, workspace_id=workspace_id)
        submitted = await repository.create_cpu_processing_task_submission(
            workspace_id=workspace_id,
            input_path=path,
            source_file=discovered,
            sha256="a" * 64,
            mime_type="image/png" if expected_kind == "image" else "text/plain",
            processing_fingerprint="cpu-task-v1",
            processor_version=3,
        )
        assert submitted.task_id is not None
        assert submitted.task_kind == expected_kind
        assert submitted.resource_class == "cpu"
        assert submitted.route_key == route_key

        async with database.session() as session:
            job = await session.get(ProcessingJob, submitted.job_id)
            source = await session.get(SourceFile, submitted.source_file_id)
            task = await session.get(VideoProcessingTask, submitted.task_id)
        assert job is not None and job.total_count == 1 and job.status == "running"
        assert source is not None and source.processing_generation == submitted.generation
        assert task is not None
        assert (task.task_kind, task.resource_class, task.route_key, task.processor_version) == (
            expected_kind,
            "cpu",
            route_key,
            3,
        )
        assert task.status == "queued" and task.owner_id is None
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cpu_batch_commit_is_fenced_and_accounts_parent_once(tmp_path: Path) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_cpu_batch_commit"
    path = tmp_path / "sample.png"
    path.write_bytes(b"cpu image task")
    discovered = DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=".png",
        size_bytes=path.stat().st_size,
    )
    try:
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
            pytest.skip("PostgreSQL integration database has not run task migration")

        repository = AssetRepository(database)
        await repository.create_workspace(name=workspace_id, workspace_id=workspace_id)
        submitted = await repository.create_cpu_processing_task_submission(
            workspace_id=workspace_id,
            input_path=path,
            source_file=discovered,
            sha256="b" * 64,
            mime_type="image/png",
            processing_fingerprint="cpu-batch-v1",
        )
        assert submitted.task_id is not None
        message = VideoTaskMessage(
            task_id=submitted.task_id,
            job_id=submitted.job_id,
            workspace_id=workspace_id,
            source_file_id=submitted.source_file_id,
            generation=submitted.generation,
            source_uri=path.as_uri(),
            task_kind=ProcessingTaskKind.IMAGE,
            resource_class=ResourceClass.CPU,
            route_key=CPU_IMAGE_ROUTE,
        )
        tasks = PostgresCpuProcessingTaskRepository(
            database.session_factory,
            task_kind=ProcessingTaskKind.IMAGE,
        )
        lease = await tasks.claim_attempt(message, worker_id="cpu-worker-1", receipt="1-0")
        assert lease is not None
        asset = AssetFactory().build_many(
            workspace_id=workspace_id,
            source_file_id=submitted.source_file_id,
            source_sha256="b" * 64,
            source_file=discovered,
            generation=submitted.generation,
            drafts=[AssetDraft(asset_type=AssetType.IMAGE, file_name=path.name)],
        )
        committer = PostgresFencedAssetBatchCommitter(database)
        async with database.session() as session, session.begin():
            source = await session.get(
                SourceFile,
                submitted.source_file_id,
                with_for_update=True,
            )
            assert source is not None
            source.sha256 = "c" * 64

        with pytest.raises(LeaseLostError, match="content is no longer current"):
            await committer.commit_assets(
                message,
                lease,
                asset,
                source_sha256="b" * 64,
            )

        async with database.session() as session:
            task = await session.get(VideoProcessingTask, submitted.task_id)
            job = await session.get(ProcessingJob, submitted.job_id)
            stored_before_retry = list(
                await session.scalars(
                    select(Asset).where(Asset.source_file_id == submitted.source_file_id)
                )
            )
        assert task is not None and task.status == "processing"
        assert task.parent_accounted_at is None
        assert job is not None and job.completed_count == 0 and job.failed_count == 0
        assert stored_before_retry == []

        async with database.session() as session, session.begin():
            source = await session.get(
                SourceFile,
                submitted.source_file_id,
                with_for_update=True,
            )
            assert source is not None
            source.sha256 = "b" * 64
        stored_ids = await committer.commit_assets(
            message,
            lease,
            asset,
            source_sha256="b" * 64,
        )
        replayed_ids = await committer.commit_assets(
            message,
            lease,
            asset,
            source_sha256="b" * 64,
        )
        assert replayed_ids == stored_ids

        async with database.session() as session:
            task = await session.get(VideoProcessingTask, submitted.task_id)
            source = await session.get(SourceFile, submitted.source_file_id)
            job = await session.get(ProcessingJob, submitted.job_id)
            stored = list(
                await session.scalars(
                    select(Asset).where(Asset.source_file_id == submitted.source_file_id)
                )
            )
        assert task is not None and task.status == "result_committed"
        assert task.parent_accounted_at is not None
        assert source is not None and source.processing_status == ProcessingStatus.COMPLETED.value
        assert job is not None and job.completed_count == 1 and job.failed_count == 0
        assert [item.asset_id for item in stored] == stored_ids
        assert await repository.list_import_job_asset_ids(job_id=submitted.job_id) == stored_ids
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()
