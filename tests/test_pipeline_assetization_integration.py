from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile, Workspace
from capsule.db.session import Database
from capsule.enums import JobStatus
from capsule.pipeline.runner import PipelineRunner


class CharacterTokenCounter:
    async def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(value) for value in texts]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_file_input_to_asset_database_with_partial_failure(tmp_path: Path) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_pipeline_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        (tmp_path / "notes.md").write_text("# Capsule\n\n正文", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("普通文本\n\n第二段", encoding="utf-8")
        Image.new("RGB", (64, 32), (1, 2, 3)).save(tmp_path / "valid.png")
        (tmp_path / "broken.png").write_bytes(b"not-an-image")

        result = await PipelineRunner(
            database=database,
            token_counter=CharacterTokenCounter(),
        ).run(tmp_path, workspace_id)

        assert result.file_count == 4
        assert result.succeeded_count == 3
        assert result.failed_count == 1
        assert result.asset_count == 5
        assert len(result.indexable_asset_ids) == 3
        assert result.skipped_count == 0
        assert result.errors[0]["relative_path"] == "broken.png"

        repeated = await PipelineRunner(
            database=database,
            token_counter=CharacterTokenCounter(),
        ).run(tmp_path, workspace_id)

        assert repeated.succeeded_count == 3
        assert repeated.failed_count == 1
        assert repeated.skipped_count == 3
        assert repeated.asset_count == 5

        async with database.session() as session:
            job = await session.get(ProcessingJob, result.job_id)
            asset_count = await session.scalar(
                select(func.count()).select_from(Asset).where(Asset.workspace_id == workspace_id)
            )
            source_count = await session.scalar(
                select(func.count())
                .select_from(SourceFile)
                .where(SourceFile.workspace_id == workspace_id)
            )
            assets = list(
                await session.scalars(
                    select(Asset)
                    .where(Asset.workspace_id == workspace_id)
                    .order_by(Asset.file_name)
                )
            )
            assert job is not None
            assert job.status == JobStatus.PARTIAL_FAILED.value
            assert job.completed_count == 3
            assert job.failed_count == 1
            assert asset_count == 5
            assert source_count == 4
            assert all(
                source.processing_status == "completed"
                for source in await session.scalars(
                    select(SourceFile).where(
                        SourceFile.workspace_id == workspace_id,
                        SourceFile.relative_path != "broken.png",
                    )
                )
            )
            assert Counter(asset.file_name for asset in assets) == {
                "notes.md": 2,
                "notes.txt": 2,
                "valid.png": 1,
            }
            markdown_assets = [asset for asset in assets if asset.file_name == "notes.md"]
            text_assets = [asset for asset in assets if asset.file_name == "notes.txt"]
            image_asset = next(asset for asset in assets if asset.file_name == "valid.png")
            assert {asset.index_role for asset in markdown_assets} == {"parent", "child"}
            assert {asset.raw_content for asset in markdown_assets} == {"# Capsule\n\n正文"}
            assert {asset.index_role for asset in text_assets} == {"parent", "child"}
            assert {asset.raw_content for asset in text_assets} == {"普通文本\n\n第二段"}
            assert image_asset.index_role == "standalone"
            assert image_asset.raw_content is None
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()
