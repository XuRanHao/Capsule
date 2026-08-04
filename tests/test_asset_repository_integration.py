from pathlib import Path

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile, Workspace
from capsule.db.repositories import (
    AssetRepository,
    EmbeddingRepository,
    LibraryClearBusyError,
    StaleAssetGenerationError,
)
from capsule.db.session import Database
from capsule.enums import (
    AssetIndexRole,
    AssetNameSource,
    AssetType,
    JobStatus,
    ProcessingStatus,
)
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.asset_factory import AssetFactory
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile


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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_assets_persists_hierarchy_and_skips_parents_for_embeddings(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_hierarchy_repository_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        path = tmp_path / "notes.md"
        path.write_text("# Notes", encoding="utf-8")
        discovered = DiscoveredFile(
            path=str(path),
            relative_path="docs/notes.md",
            extension=".md",
            size_bytes=path.stat().st_size,
        )
        repository = AssetRepository(database)
        source_file_id = await repository.get_or_create_source_file(
            workspace_id=workspace_id,
            source_file=discovered,
            sha256=sha256_file(path),
            mime_type="text/markdown",
        )
        factory = AssetFactory()

        def build_hierarchy(child_positions: list[tuple[int, int]]) -> list[AssetCreate]:
            return factory.build_many(
                workspace_id=workspace_id,
                source_file_id=source_file_id,
                source_sha256=sha256_file(path),
                source_file=discovered,
                drafts=[
                    AssetDraft(
                        asset_type=AssetType.MARKDOWN_BLOCK,
                        file_name=path.name,
                        index_role=AssetIndexRole.PARENT,
                        hierarchy_key="section:root",
                        source_locator={"block_index": 0},
                        raw_content="Section summary",
                    ),
                    *[
                        AssetDraft(
                            asset_type=AssetType.MARKDOWN_BLOCK,
                            file_name=path.name,
                            parent_hierarchy_key="section:root",
                            child_order=child_order,
                            source_locator={"block_index": block_index},
                            raw_content=f"Section detail {block_index}",
                        )
                        for block_index, child_order in child_positions
                    ],
                ],
            )

        assets = build_hierarchy([(1, 0)])

        stored = await repository.replace_assets(source_file_id=source_file_id, assets=assets)

        assert stored.asset_ids == [asset.asset_id for asset in assets]
        assert stored.indexable_asset_ids == [assets[1].asset_id]
        async with database.session() as session:
            rows = {
                asset.asset_id: asset
                for asset in await session.scalars(
                    select(Asset).where(Asset.source_file_id == source_file_id)
                )
            }
        parent, child = assets
        assert rows[parent.asset_id].index_role == AssetIndexRole.PARENT.value
        assert rows[parent.asset_id].processing_status == ProcessingStatus.COMPLETED.value
        assert rows[child.asset_id].index_role == AssetIndexRole.CHILD.value
        assert rows[child.asset_id].parent_asset_id == parent.asset_id
        assert rows[child.asset_id].child_order == 0

        embedding_assets = await EmbeddingRepository(database).list_assets(
            workspace_id=workspace_id,
            asset_ids=stored.asset_ids,
        )
        assert [asset.asset_id for asset in embedding_assets] == [child.asset_id]

        # A new child can reuse a stale child's order under the stable parent.
        replacement = build_hierarchy([(2, 0)])
        replacement_stored = await repository.replace_assets(
            source_file_id=source_file_id,
            assets=replacement,
        )
        replacement_child_id = replacement_stored.asset_ids[1]
        assert replacement_stored.asset_ids[0] == parent.asset_id
        assert replacement_child_id != child.asset_id
        async with database.session() as session:
            assert await session.get(Asset, child.asset_id) is None
            replacement_child = await session.get(Asset, replacement_child_id)
            assert replacement_child is not None
            assert replacement_child.parent_asset_id == parent.asset_id
            assert replacement_child.child_order == 0

        # Two stable children can swap orders in one deferred-constraint transaction.
        swap_seed = build_hierarchy([(2, 0), (3, 1)])
        swap_seed_stored = await repository.replace_assets(
            source_file_id=source_file_id,
            assets=swap_seed,
        )
        swapped = build_hierarchy([(2, 1), (3, 0)])
        swapped_stored = await repository.replace_assets(
            source_file_id=source_file_id,
            assets=swapped,
        )
        assert swapped_stored.asset_ids == swap_seed_stored.asset_ids
        async with database.session() as session:
            first_child = await session.get(Asset, swapped_stored.asset_ids[1])
            second_child = await session.get(Asset, swapped_stored.asset_ids[2])
            assert first_child is not None
            assert second_child is not None
            assert first_child.child_order == 1
            assert second_child.child_order == 0
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
async def test_incremental_generation_is_idempotent_and_rejects_stale_delivery(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_generation_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        repository = AssetRepository(database)
        path = tmp_path / "video.mp4"
        path.write_bytes(b"generation-one")
        discovered = DiscoveredFile(
            path=str(path),
            relative_path=path.name,
            extension=".mp4",
            size_bytes=path.stat().st_size,
        )
        first = await repository.prepare_source_file(
            workspace_id=workspace_id,
            source_file=discovered,
            sha256="1" * 64,
            mime_type="video/mp4",
            processing_fingerprint="a" * 64,
        )
        factory = AssetFactory()
        drafts = [
            AssetDraft(
                asset_type=AssetType.VIDEO_SEGMENT,
                file_name=path.name,
                source_locator={"start_ms": index * 1_000, "end_ms": (index + 1) * 1_000},
                file_info={"fps": 30.0},
            )
            for index in range(2)
        ]
        first_assets = factory.build_many(
            workspace_id=workspace_id,
            source_file_id=first.source_file_id,
            source_sha256="1" * 64,
            source_file=discovered,
            drafts=drafts,
            generation=first.generation,
        )
        first_id = await repository.upsert_generated_asset(
            source_file_id=first.source_file_id,
            generation=first.generation,
            asset=first_assets[0],
        )
        duplicate_id = await repository.upsert_generated_asset(
            source_file_id=first.source_file_id,
            generation=first.generation,
            asset=first_assets[0],
        )
        await repository.upsert_generated_asset(
            source_file_id=first.source_file_id,
            generation=first.generation,
            asset=first_assets[1],
        )
        assert duplicate_id == first_id

        second = await repository.prepare_source_file(
            workspace_id=workspace_id,
            source_file=discovered,
            sha256="2" * 64,
            mime_type="video/mp4",
            processing_fingerprint="b" * 64,
        )
        with pytest.raises(StaleAssetGenerationError):
            await repository.upsert_generated_asset(
                source_file_id=first.source_file_id,
                generation=first.generation,
                asset=first_assets[0],
            )

        second_asset = factory.build_many(
            workspace_id=workspace_id,
            source_file_id=second.source_file_id,
            source_sha256="2" * 64,
            source_file=discovered,
            drafts=drafts[:1],
            generation=second.generation,
        )[0]
        await repository.upsert_generated_asset(
            source_file_id=second.source_file_id,
            generation=second.generation,
            asset=second_asset,
        )
        visible = await repository.list_asset_views(workspace_id=workspace_id)
        assert visible.total == 1
        assert await repository.finalize_asset_generation_if_complete(
            source_file_id=second.source_file_id,
            generation=second.generation,
            expected_asset_count=1,
        )

        async with database.session() as session:
            rows = list(
                await session.scalars(
                    select(Asset).where(Asset.source_file_id == second.source_file_id)
                )
            )
            source_row = await session.get(SourceFile, second.source_file_id)
        assert len(rows) == 1
        assert rows[0].generation == second.generation
        assert source_row is not None
        assert source_row.processing_status == ProcessingStatus.COMPLETED.value
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
async def test_repository_clear_library_rejects_active_jobs(
    tmp_path: Path,
) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_clear_repository_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        repository = AssetRepository(database)
        job_id = await repository.create_job(
            workspace_id=workspace_id,
            input_path=tmp_path,
            total_count=1,
        )
        path = tmp_path / "clear.md"
        path.write_text("# clear", encoding="utf-8")
        source_file_id = await repository.get_or_create_source_file(
            workspace_id=workspace_id,
            source_file=DiscoveredFile(
                path=str(path),
                relative_path=path.name,
                extension=".md",
                size_bytes=path.stat().st_size,
            ),
            sha256=sha256_file(path),
            mime_type="text/markdown",
        )

        with pytest.raises(LibraryClearBusyError):
            await repository.clear_all_records()

        async with database.session() as session:
            assert await session.get(Workspace, workspace_id) is not None
            assert await session.get(SourceFile, source_file_id) is not None
            assert await session.get(ProcessingJob, job_id) is not None
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()
