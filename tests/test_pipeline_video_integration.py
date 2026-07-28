import asyncio
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, Workspace
from capsule.db.session import Database
from capsule.pipeline.runner import PipelineRunner

VIDEO_FIXTURE = Path("data/dev-fixtures/nature/hiking-trip.mp4").resolve()


class FakeVideoEmbedder:
    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), 1.0] for index, _ in enumerate(frames)],
            dtype=np.float32,
        )


def _create_png(path: Path) -> None:
    Image.new("RGB", (32, 32), (10, 20, 30)).save(path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_video_file_is_stored_as_logical_segment_assets(tmp_path: Path) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_video_pipeline_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        video = tmp_path / VIDEO_FIXTURE.name
        await asyncio.to_thread(shutil.copyfile, VIDEO_FIXTURE, video)

        result = await PipelineRunner(
            database=database,
            video_embedder=FakeVideoEmbedder(),
        ).run(tmp_path, workspace_id)

        assert result.succeeded_count == 1
        assert result.failed_count == 0
        assert result.asset_count >= 1
        async with database.session() as session:
            assets = list(
                await session.scalars(
                    select(Asset).where(Asset.workspace_id == workspace_id).order_by(Asset.asset_id)
                )
            )
            assert assets
            assert all(asset.asset_type == "video_segment" for asset in assets)
            assert all(asset.raw_content is None for asset in assets)
            assert all(asset.source_locator["type"] == "time_range" for asset in assets)
    finally:
        async with database.session() as session, session.begin():
            workspace = await session.get(Workspace, workspace_id, with_for_update=True)
            if workspace is not None:
                await session.delete(workspace)
        await database.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_mps_records_only_the_video_as_failed(tmp_path: Path) -> None:
    database = Database(get_settings())
    workspace_id = "workspace_video_mps_failure_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        await asyncio.to_thread(shutil.copyfile, VIDEO_FIXTURE, tmp_path / VIDEO_FIXTURE.name)
        await asyncio.to_thread(_create_png, tmp_path / "still.png")

        result = await PipelineRunner(database=database).run(tmp_path, workspace_id)

        assert result.succeeded_count == 1
        assert result.failed_count == 1
        assert result.asset_count == 1
        assert result.errors[0]["relative_path"] == VIDEO_FIXTURE.name
        assert "MobileCLIP" in result.errors[0]["error"]
    finally:
        async with database.session() as session, session.begin():
            workspace = await session.get(Workspace, workspace_id, with_for_update=True)
            if workspace is not None:
                await session.delete(workspace)
        await database.dispose()
