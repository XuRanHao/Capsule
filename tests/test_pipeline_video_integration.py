import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

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


class FakeObjectStorage:
    def __init__(self) -> None:
        self.bucket_ready = False
        self.objects: dict[str, bytes] = {}
        self.content_types: dict[str, str | None] = {}

    async def ensure_bucket(self) -> None:
        self.bucket_ready = True

    async def upload_file(
        self,
        source: Path,
        object_key: str,
        *,
        content_type: str | None = None,
    ) -> str:
        self.objects[object_key] = await asyncio.to_thread(source.read_bytes)
        self.content_types[object_key] = content_type
        return f"s3://capsule/{object_key}"


def _create_png(path: Path) -> None:
    Image.new("RGB", (32, 32), (10, 20, 30)).save(path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_video_file_stores_playable_segment_media(tmp_path: Path) -> None:
    database = Database(get_settings())
    workspace_id = f"workspace_video_{uuid4().hex[:12]}"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        video = tmp_path / VIDEO_FIXTURE.name
        await asyncio.to_thread(shutil.copyfile, VIDEO_FIXTURE, video)

        storage = FakeObjectStorage()
        result = await PipelineRunner(
            database=database,
            video_embedder=FakeVideoEmbedder(),
            object_storage=storage,
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
            assert all(asset.derived_file_uri for asset in assets)
            assert all(asset.preview_uri for asset in assets)
            assert all(asset.file_info["keyframes"] for asset in assets)
            assert all(
                all(
                    frame["uri"].startswith("s3://capsule/derived/video-segments/")
                    for frame in asset.file_info["keyframes"]
                )
                for asset in assets
            )
        assert storage.bucket_ready
        assert any(content[4:8] == b"ftyp" for content in storage.objects.values())
        assert any(content.startswith(b"\xff\xd8") for content in storage.objects.values())
        assert set(storage.content_types.values()) == {"video/mp4", "image/jpeg"}
        stored_objects = dict(storage.objects)

        repeated = await PipelineRunner(
            database=database,
            video_embedder=FakeVideoEmbedder(),
            object_storage=storage,
        ).run(tmp_path, workspace_id)

        assert repeated.succeeded_count == 1
        assert repeated.failed_count == 0
        assert repeated.skipped_count == 1
        assert repeated.asset_count == result.asset_count
        assert storage.objects == stored_objects
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
    workspace_id = f"workspace_video_mps_{uuid4().hex[:12]}"
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
