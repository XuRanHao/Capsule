import asyncio
import threading
import time
from pathlib import Path

import pytest

from capsule.enums import AssetType
from capsule.pipeline import video_media
from capsule.pipeline.video_media import VideoDerivedMediaWriter
from capsule.schemas import AssetCreate, DiscoveredFile


def test_render_artifacts_seeks_before_opening_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    asset = AssetCreate(
        asset_id="asset_test",
        workspace_id="workspace_test",
        source_file_id="source_test",
        asset_type=AssetType.VIDEO_SEGMENT,
        file_name=source.name,
        file_type=".mp4",
        asset_key="segment-test",
        content_hash="a" * 64,
        source_locator={"start_ms": 1_250, "end_ms": 2_250},
        file_info={"fps": 30.0, "representative_frames": [{"timestamp_ms": 1_750}]},
    )
    commands: list[list[str]] = []

    def fake_run_ffmpeg(_ffmpeg: Path, arguments: list[str]) -> None:
        commands.append(arguments)
        output = tmp_path / "output"
        (output / "segment.mp4").write_bytes(b"segment")
        (output / "keyframe-01.jpg").write_bytes(b"keyframe")

    monkeypatch.setattr(video_media, "resolve_video_tool", lambda _name: Path("ffmpeg"))
    monkeypatch.setattr(
        video_media,
        "_video_encoder_arguments",
        lambda _ffmpeg: ("fake_encoder", ()),
    )
    monkeypatch.setattr(video_media, "_run_ffmpeg", fake_run_ffmpeg)

    video_media._render_artifacts(source, asset, tmp_path / "output", True)

    assert len(commands) == 1
    assert commands[0][:7] == [
        "-y",
        "-ss",
        "1.250",
        "-i",
        str(source),
        "-i",
        str(source),
    ]
    assert any("[1:a:0]atrim=start=1.250:duration=1.000" in argument for argument in commands[0])
    assert any("select='eq(n\\,15)'" in argument for argument in commands[0])
    assert "[segment_video]" in commands[0]
    assert "[keyframes]" in commands[0]
    assert "[segment_audio]" in commands[0]


def test_render_artifacts_does_not_require_audio_stream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"video")
    asset = AssetCreate(
        asset_id="asset_silent",
        workspace_id="workspace_test",
        source_file_id="source_test",
        asset_type=AssetType.VIDEO_SEGMENT,
        file_name=source.name,
        file_type=".mp4",
        asset_key="segment-silent",
        content_hash="b" * 64,
        source_locator={"start_ms": 0, "end_ms": 1_000},
        file_info={"fps": 30.0, "representative_frames": [{"timestamp_ms": 500}]},
    )
    commands: list[list[str]] = []

    def fake_run_ffmpeg(_ffmpeg: Path, arguments: list[str]) -> None:
        commands.append(arguments)
        output = tmp_path / "output"
        (output / "segment.mp4").write_bytes(b"segment")
        (output / "keyframe-01.jpg").write_bytes(b"keyframe")

    monkeypatch.setattr(video_media, "resolve_video_tool", lambda _name: Path("ffmpeg"))
    monkeypatch.setattr(
        video_media,
        "_video_encoder_arguments",
        lambda _ffmpeg: ("fake_encoder", ()),
    )
    monkeypatch.setattr(video_media, "_run_ffmpeg", fake_run_ffmpeg)

    video_media._render_artifacts(source, asset, tmp_path / "output", False)

    assert commands[0].count("-i") == 1
    assert "[segment_audio]" not in commands[0]


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
        _source_has_audio: bool,
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
                "fps": 30.0,
                "representative_frames": [
                    {"timestamp_ms": index * 1000},
                    {"timestamp_ms": index * 1000 + 500},
                ],
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
    monkeypatch.setattr(video_media, "_source_has_audio", lambda _source: True)
    monkeypatch.setattr(video_media, "_render_artifacts", fake_render)
    writer = VideoDerivedMediaWriter(storage, concurrency=2)

    first, second = await asyncio.gather(
        writer.persist(source_file=discovered, assets=assets[:2]),
        writer.persist(source_file=discovered, assets=assets[2:]),
    )
    await writer.close()

    assert max_active == 2
    assert storage.ensure_calls == 1
    assert storage.upload_calls == 16
    assert all(asset.derived_file_uri for asset in [*first, *second])
    assert all(asset.preview_uri for asset in [*first, *second])
    assert all(len(asset.file_info["keyframes"]) == 2 for asset in [*first, *second])


async def test_ffmpeg_slot_is_released_while_upload_is_blocked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    rendered = 0
    both_rendered = asyncio.Event()
    upload_gate = asyncio.Event()

    def fake_render(
        _source: Path,
        _asset: AssetCreate,
        output_directory: Path,
        _source_has_audio: bool,
    ) -> video_media._RenderedArtifacts:
        nonlocal rendered
        output_directory.mkdir(parents=True)
        segment = output_directory / "segment.mp4"
        frame = output_directory / "frame.jpg"
        segment.write_bytes(b"segment")
        frame.write_bytes(b"frame")
        rendered += 1
        if rendered == 2:
            both_rendered.set()
        return video_media._RenderedArtifacts(
            segment_path=segment,
            keyframe_paths=[frame],
            video_encoder="fake",
        )

    class BlockingStorage:
        async def ensure_bucket(self) -> None:
            return None

        async def upload_file(
            self,
            _source: Path,
            object_key: str,
            *,
            content_type: str | None = None,
        ) -> str:
            del content_type
            await upload_gate.wait()
            return f"s3://bucket/{object_key}"

    assets = [_video_asset(source, index) for index in range(2)]
    discovered = _discovered_video(source)
    monkeypatch.setattr(video_media, "_source_has_audio", lambda _source: False)
    monkeypatch.setattr(video_media, "_render_artifacts", fake_render)
    writer = VideoDerivedMediaWriter(
        BlockingStorage(),
        concurrency=1,
        upload_concurrency=1,
        spool_root=tmp_path / "spool",
    )
    task = asyncio.create_task(writer.persist(source_file=discovered, assets=assets))
    try:
        await asyncio.wait_for(both_rendered.wait(), timeout=1)
        assert not task.done()
        upload_gate.set()
        persisted = await task
        assert len(persisted) == 2
    finally:
        upload_gate.set()
        await writer.close()


async def test_upload_failure_retries_without_duplicate_persist_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    def fake_render(
        _source: Path,
        _asset: AssetCreate,
        output_directory: Path,
        _source_has_audio: bool,
    ) -> video_media._RenderedArtifacts:
        output_directory.mkdir(parents=True)
        segment = output_directory / "segment.mp4"
        frame = output_directory / "frame.jpg"
        segment.write_bytes(b"segment")
        frame.write_bytes(b"frame")
        return video_media._RenderedArtifacts(
            segment_path=segment,
            keyframe_paths=[frame],
            video_encoder="fake",
        )

    class FlakyStorage:
        def __init__(self) -> None:
            self.failed = False
            self.calls = 0

        async def ensure_bucket(self) -> None:
            return None

        async def upload_file(
            self,
            _source: Path,
            object_key: str,
            *,
            content_type: str | None = None,
        ) -> str:
            del content_type
            self.calls += 1
            if object_key.endswith("segment.mp4") and not self.failed:
                self.failed = True
                raise RuntimeError("transient upload failure")
            return f"s3://bucket/{object_key}"

    monkeypatch.setattr(video_media, "_source_has_audio", lambda _source: False)
    monkeypatch.setattr(video_media, "_render_artifacts", fake_render)
    storage = FlakyStorage()
    callbacks: list[str] = []

    async def persisted(asset: AssetCreate, expected_count: int) -> str:
        assert expected_count == 1
        callbacks.append(asset.asset_id)
        return asset.asset_id

    writer = VideoDerivedMediaWriter(
        storage,
        concurrency=1,
        upload_concurrency=1,
        spool_root=tmp_path / "spool",
        max_upload_attempts=3,
        retry_base_seconds=0,
        on_asset_persisted=persisted,
    )
    try:
        result = await writer.persist(
            source_file=_discovered_video(source),
            assets=[_video_asset(source, 0)],
        )
    finally:
        await writer.close()

    assert result[0].derived_file_uri is not None
    assert callbacks == ["asset_0"]
    assert storage.calls > 3
    assert list((tmp_path / "spool").iterdir()) == []


async def test_stale_generation_is_rejected_before_upload_without_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")

    def fake_render(
        _source: Path,
        _asset: AssetCreate,
        output_directory: Path,
        _source_has_audio: bool,
    ) -> video_media._RenderedArtifacts:
        output_directory.mkdir(parents=True)
        segment = output_directory / "segment.mp4"
        frame = output_directory / "frame.jpg"
        segment.write_bytes(b"segment")
        frame.write_bytes(b"frame")
        return video_media._RenderedArtifacts(
            segment_path=segment,
            keyframe_paths=[frame],
            video_encoder="fake",
        )

    class UnexpectedStorage:
        async def ensure_bucket(self) -> None:
            return None

        async def upload_file(
            self,
            _source: Path,
            _object_key: str,
            *,
            content_type: str | None = None,
        ) -> str:
            del content_type
            raise AssertionError("stale work must not reach object storage")

    validations = 0

    async def reject_stale(_asset: AssetCreate) -> None:
        nonlocal validations
        validations += 1
        raise video_media.ObsoleteVideoUploadError("generation is obsolete")

    monkeypatch.setattr(video_media, "_source_has_audio", lambda _source: False)
    monkeypatch.setattr(video_media, "_render_artifacts", fake_render)
    writer = VideoDerivedMediaWriter(
        UnexpectedStorage(),
        spool_root=tmp_path / "spool",
        max_upload_attempts=4,
        validate_asset_generation=reject_stale,
    )
    try:
        with pytest.raises(video_media.ObsoleteVideoUploadError):
            await writer.persist(
                source_file=_discovered_video(source),
                assets=[_video_asset(source, 0)],
            )
    finally:
        await writer.close()

    assert validations == 1
    assert list((tmp_path / "spool").iterdir()) == []


async def test_complete_spool_manifest_is_republished_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    spool_root = tmp_path / "spool"
    bundle = spool_root / "recovered-bundle"
    bundle.mkdir(parents=True)
    segment = bundle / "segment.mp4"
    frame = bundle / "frame.jpg"
    segment.write_bytes(b"segment")
    frame.write_bytes(b"frame")
    asset = _video_asset(source, 0)
    manifest = video_media._UploadManifest(
        manifest_id="recovered-manifest",
        asset=asset,
        segment_path=segment.name,
        keyframe_paths=[frame.name],
        video_encoder="fake",
        spool_bytes=segment.stat().st_size + frame.stat().st_size,
        generation_asset_count=1,
    )
    (bundle / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    committed = asyncio.Event()

    class Storage:
        async def ensure_bucket(self) -> None:
            return None

        async def upload_file(
            self,
            _source: Path,
            object_key: str,
            *,
            content_type: str | None = None,
        ) -> str:
            del content_type
            return f"s3://bucket/{object_key}"

    async def on_asset_persisted(asset: AssetCreate, expected_count: int) -> str:
        assert expected_count == 1
        committed.set()
        return asset.asset_id

    writer = VideoDerivedMediaWriter(
        Storage(),
        spool_root=spool_root,
        on_asset_persisted=on_asset_persisted,
    )

    try:
        await writer.start()
        assert committed.is_set()
    finally:
        await writer.close()

    assert list(spool_root.iterdir()) == []


def _video_asset(source: Path, index: int) -> AssetCreate:
    return AssetCreate(
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
            "fps": 30.0,
            "representative_frames": [{"timestamp_ms": index * 1000 + 500}],
        },
    )


def _discovered_video(source: Path) -> DiscoveredFile:
    return DiscoveredFile(
        path=str(source),
        relative_path=source.name,
        extension=".mp4",
        size_bytes=source.stat().st_size,
    )
