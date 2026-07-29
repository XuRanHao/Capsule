from pathlib import Path

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, Workspace
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.enums import AssetNameSource, AssetType, JobStatus
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.asset_factory import AssetFactory
from capsule.schemas import AssetDraft, DiscoveredFile


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repository_replaces_assets_and_preserves_user_name(tmp_path: Path) -> None:
    database = Database(get_settings())
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        workspace_id = "workspace_repository_integration"
        repository = AssetRepository(database)
        path = tmp_path / "notes.md"
        path.write_text("# First", encoding="utf-8")
        discovered = DiscoveredFile(
            path=str(path),
            relative_path="docs/notes.md",
            extension=".md",
            size_bytes=path.stat().st_size,
        )
        job_id = await repository.create_job(
            workspace_id=workspace_id,
            input_path=tmp_path,
            total_count=1,
        )
        source_file_id = await repository.get_or_create_source_file(
            workspace_id=workspace_id,
            source_file=discovered,
            sha256=sha256_file(path),
            mime_type="text/markdown",
        )
        factory = AssetFactory()
        locator = {"type": "text_range", "block_index": 0, "char_start": 0, "char_end": 7}
        first = factory.build_many(
            workspace_id=workspace_id,
            source_file_id=source_file_id,
            source_sha256=sha256_file(path),
            source_file=discovered,
            drafts=[
                AssetDraft(
                    asset_type=AssetType.MARKDOWN_BLOCK,
                    file_name=path.name,
                    source_locator=locator,
                    raw_content="# First",
                )
            ],
        )
        first_result = await repository.replace_assets(
            source_file_id=source_file_id,
            assets=first,
        )

        async with database.session() as session, session.begin():
            asset = await session.scalar(
                select(Asset).where(Asset.asset_id == first_result.asset_ids[0])
            )
            assert asset is not None
            asset.asset_name = "人工标题"
            asset.asset_name_source = AssetNameSource.USER.value
            asset.asset_description = "旧模型描述"
            asset.asset_features = {"subject_content": {"value": "旧内容"}}

        path.write_text("# Second", encoding="utf-8")
        discovered.size_bytes = path.stat().st_size
        second = factory.build_many(
            workspace_id=workspace_id,
            source_file_id=source_file_id,
            source_sha256=sha256_file(path),
            source_file=discovered,
            drafts=[
                AssetDraft(
                    asset_type=AssetType.MARKDOWN_BLOCK,
                    file_name=path.name,
                    source_locator=locator,
                    raw_content="# Second",
                )
            ],
        )
        second_result = await repository.replace_assets(
            source_file_id=source_file_id,
            assets=second,
        )
        await repository.add_job_stage_durations(
            job_id=job_id,
            durations_ms={"parsing": 125.5, "asset_stored": 24.5},
        )
        await repository.record_file_success(job_id=job_id)
        await repository.finalize_job(job_id=job_id)

        assert second_result.asset_ids == first_result.asset_ids
        async with database.session() as session:
            asset = await session.get(Asset, second_result.asset_ids[0])
            job = await session.get(ProcessingJob, job_id)
            assert asset is not None
            assert job is not None
            assert asset.raw_content == "# Second"
            assert asset.file_tree_context == ["docs"]
            assert asset.asset_name == "人工标题"
            assert asset.asset_name_source == AssetNameSource.USER.value
            assert asset.asset_description is None
            assert asset.asset_features == {}
            assert asset.feature_revision == 2
            assert asset.embedding_revision == 2
            assert job.status == JobStatus.COMPLETED.value
            assert job.completed_count == 1
            assert job.failed_count == 0
            assert job.stage_durations_ms == {
                "parsing": 125.5,
                "asset_stored": 24.5,
            }
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(
                        Workspace.workspace_id == "workspace_repository_integration"
                    )
                )
        finally:
            await database.dispose()
