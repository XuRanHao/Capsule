import asyncio
import threading
import time
from pathlib import Path

from capsule.enums import AssetType
from capsule.pipeline import video_media
from capsule.pipeline.video_media import VideoDerivedMediaWriter
from capsule.schemas import AssetCreate, DiscoveredFile


async def test_video_media_writer_bounds_work_and_initializes_storage_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    def fake_render(
        _source: Path,
        _asset: AssetCreate,
        output_directory: Path,
    ) -> video_media._RenderedArtifacts:
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            output_directory.mkdir(parents=True)
            segment = output_directory / "segment.mp4"
            first = output_directory / "first.jpg"
            second = output_directory / "second.jpg"
            segment.write_bytes(b"segment")
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            return video_media._RenderedArtifacts(
                segment_path=segment,
                keyframe_paths=[first, second],
                video_encoder="fake",
            )
        finally:
            with counter_lock:
                active -= 1

    class FakeStorage:
        def __init__(self) -> None:
            self.ensure_calls = 0
            self.upload_calls = 0

        async def ensure_bucket(self) -> None:
            self.ensure_calls += 1
            await asyncio.sleep(0.005)

        async def upload_file(
            self,
            _source: Path,
            object_key: str,
            *,
            content_type: str | None = None,
        ) -> str:
            del content_type
            self.upload_calls += 1
            await asyncio.sleep(0.005)
            return f"s3://bucket/{object_key}"

    assets = [
        AssetCreate(
            asset_id=f"asset_{index}",
            workspace_id="workspace_test",
            source_file_id="source_test",
            asset_type=AssetType.VIDEO_SEGMENT,
            file_name=source.name,
            file_type=".mp4",
            asset_key=f"segment-{index}",
            content_hash=str(index) * 64,
            source_locator={"start_ms": index * 1000, "end_ms": (index + 1) * 1000},
            file_info={
                "representative_frames": [
                    {"timestamp_ms": index * 1000},
                    {"timestamp_ms": index * 1000 + 500},
                ]
            },
        )
        for index in range(4)
    ]
    discovered = DiscoveredFile(
        path=str(source),
        relative_path=source.name,
        extension=".mp4",
        size_bytes=source.stat().st_size,
    )
    storage = FakeStorage()
    monkeypatch.setattr(video_media, "_render_artifacts", fake_render)
    writer = VideoDerivedMediaWriter(storage, concurrency=2)

    first, second = await asyncio.gather(
        writer.persist(source_file=discovered, assets=assets[:2]),
        writer.persist(source_file=discovered, assets=assets[2:]),
    )

    assert max_active == 2
    assert storage.ensure_calls == 1
    assert storage.upload_calls == 16
    assert all(asset.derived_file_uri for asset in [*first, *second])
    assert all(asset.preview_uri for asset in [*first, *second])
    assert all(len(asset.file_info["keyframes"]) == 2 for asset in [*first, *second])
