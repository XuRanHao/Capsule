import asyncio
import signal
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from PIL import Image

from capsule.api import assets as asset_api
from capsule.api.app import create_app
from capsule.config import Settings
from capsule.db.repositories import AssetMediaTarget
from capsule.enums import AssetType
from capsule.schemas import (
    AssetListResponse,
    AssetSourceRecord,
    AssetViewRecord,
    LibraryClearResult,
)
from capsule.storage.object_storage import ObjectDownload, ObjectStorage


class FakeAssetRepository:
    def __init__(self, image_path: Path, *, derived_image_path: Path | None = None) -> None:
        self.image_path = image_path
        self.derived_image_path = derived_image_path
        now = datetime.now(UTC)
        self.asset = AssetViewRecord(
            asset_id="asset_image",
            workspace_id="workspace_test",
            project_id="project_test",
            source_file_id="source_image",
            asset_type=AssetType.IMAGE,
            file_name="image.jpg",
            file_type="jpg",
            asset_name="测试图片",
            asset_description="一张真实接口返回的测试图片",
            asset_features={},
            file_tree_context=[],
            source_contexts=[],
            file_info={"width": 1, "height": 1},
            source_locator={"kind": "whole_file"},
            processing_status="completed",
            feature_revision=1,
            embedding_revision=1,
            source_file=AssetSourceRecord(
                source_file_id="source_image",
                original_file_name="image.jpg",
                relative_path="folder/image.jpg",
                file_type="jpg",
                mime_type="image/jpeg",
                file_size_bytes=image_path.stat().st_size,
                processing_status="completed",
            ),
            created_at=now,
            updated_at=now,
        )

    async def list_asset_views(self, **_: object) -> AssetListResponse:
        return AssetListResponse(items=[self.asset], total=1, limit=100, offset=0)

    async def get_asset_view(self, **_: object) -> AssetViewRecord:
        return self.asset

    async def get_asset_media(self, **_: object) -> AssetMediaTarget:
        return AssetMediaTarget(
            asset_id=self.asset.asset_id,
            workspace_id=self.asset.workspace_id,
            asset_type=self.asset.asset_type.value,
            source_storage_uri=self.image_path.as_uri(),
            source_mime_type="image/jpeg",
            preview_uri=None,
            derived_file_uri=(
                self.derived_image_path.as_uri() if self.derived_image_path is not None else None
            ),
        )


class FakeLibraryClearService:
    def __init__(self) -> None:
        self.clear_calls = 0

    async def clear_all(self) -> LibraryClearResult:
        self.clear_calls += 1
        return LibraryClearResult(
            workspaces_deleted=2,
            assets_deleted=3,
            source_files_deleted=1,
            embeddings_deleted=2,
            jobs_deleted=1,
            vectors_deleted=2,
            objects_deleted=4,
            staging_paths_deleted=1,
        )


class FakeVideoAssetRepository(FakeAssetRepository):
    async def get_asset_media(self, **_: object) -> AssetMediaTarget:
        return AssetMediaTarget(
            asset_id="asset_video",
            workspace_id="workspace_test",
            asset_type=AssetType.VIDEO_SEGMENT.value,
            source_storage_uri="s3://capsule/source/video.mp4",
            source_mime_type="video/mp4",
            preview_uri="s3://capsule/derived/preview.jpg",
            derived_file_uri="s3://capsule/derived/segment.mp4",
        )


class FakeLogicalVideoAssetRepository:
    def __init__(
        self,
        source_path: Path,
        *,
        source_mime_type: str,
        locator: dict[str, object],
        derived_file_uri: str | None = None,
        preview_uri: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        self.source_path = source_path
        self.source_mime_type = source_mime_type
        self.derived_file_uri = derived_file_uri
        self.preview_uri = preview_uri
        self.asset = AssetViewRecord(
            asset_id="asset_video",
            workspace_id="workspace_test",
            project_id="project_test",
            source_file_id="source_video",
            asset_type=AssetType.VIDEO_SEGMENT,
            file_name=source_path.name,
            file_type=source_path.suffix,
            asset_features={},
            file_tree_context=[],
            source_contexts=[],
            file_info={},
            source_locator=locator,
            source_storage_uri=source_path.as_uri(),
            processing_status="completed",
            feature_revision=1,
            embedding_revision=1,
            source_file=AssetSourceRecord(
                source_file_id="source_video",
                original_file_name=source_path.name,
                relative_path=source_path.name,
                file_type=source_path.suffix,
                mime_type=source_mime_type,
                file_size_bytes=source_path.stat().st_size,
                processing_status="completed",
            ),
            derived_file_uri=derived_file_uri,
            preview_uri=preview_uri,
            created_at=now,
            updated_at=now,
        )

    async def list_asset_views(self, **_: object) -> AssetListResponse:
        return AssetListResponse(items=[self.asset], total=1, limit=100, offset=0)

    async def get_asset_view(self, **_: object) -> AssetViewRecord:
        return self.asset

    async def get_asset_media(self, **_: object) -> AssetMediaTarget:
        return AssetMediaTarget(
            asset_id=self.asset.asset_id,
            workspace_id=self.asset.workspace_id,
            asset_type=self.asset.asset_type.value,
            source_storage_uri=self.source_path.as_uri(),
            source_mime_type=self.source_mime_type,
            preview_uri=self.preview_uri,
            derived_file_uri=self.derived_file_uri,
        )

def test_asset_list_and_local_preview_are_available(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    image_path = import_root / "image.jpg"
    Image.new("RGB", (1200, 800), "#e05b3f").save(image_path, quality=95)
    repository = FakeAssetRepository(image_path)
    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=repository,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/assets",
            params={"workspace_id": "workspace_test"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["items"][0]["asset_name"] == "测试图片"
        assert payload["items"][0]["preview_url"] == (
            "/api/v1/assets/asset_image/thumbnail?workspace_id=workspace_test"
        )
        assert payload["items"][0]["content_url"] == (
            "/api/v1/assets/asset_image/content?workspace_id=workspace_test"
        )

        preview = client.get(
            "/api/v1/assets/asset_image/preview",
            params={"workspace_id": "workspace_test"},
        )
        thumbnail = client.get(
            "/api/v1/assets/asset_image/thumbnail",
            params={"workspace_id": "workspace_test"},
        )
        assert preview.status_code == 200
        assert preview.content == image_path.read_bytes()
        assert thumbnail.status_code == 200
        assert thumbnail.headers["content-type"] == "image/jpeg"
        assert thumbnail.headers["cache-control"] == "public, max-age=86400, immutable"
        with Image.open(BytesIO(thumbnail.content)) as rendered:
            assert max(rendered.size) == 480


def test_library_clear_endpoint_is_permanently_retired(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    image_path = import_root / "image.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    clear_service = FakeLibraryClearService()
    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=FakeAssetRepository(image_path),  # type: ignore[arg-type]
        library_clear_service=clear_service,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        retired = client.post(
            "/api/v1/assets/clear-all",
            json={"confirmation": "CLEAR ALL DATA"},
        )
        assert retired.status_code == 410
        assert retired.json()["detail"]["code"] == "library_clear_retired"
        assert clear_service.clear_calls == 0


def test_document_image_preview_uses_derived_media_root(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    document_path = import_root / "report.docx"
    document_path.write_bytes(b"docx-container")
    media_root = tmp_path / "document-media"
    media_root.mkdir()
    image_path = media_root / "embedded.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    app = create_app(
        settings=Settings(import_root=import_root, document_media_root=media_root),
        asset_repository=FakeAssetRepository(  # type: ignore[arg-type]
            document_path,
            derived_image_path=image_path,
        ),
    )

    with TestClient(app) as client:
        preview = client.get(
            "/api/v1/assets/asset_image/preview",
            params={"workspace_id": "workspace_test"},
        )
        content = client.get(
            "/api/v1/assets/asset_image/content",
            params={"workspace_id": "workspace_test"},
        )

    assert preview.status_code == 200
    assert content.status_code == 200
    assert preview.content == image_path.read_bytes()
    assert content.content == image_path.read_bytes()


def test_s3_video_media_is_proxied_with_thumbnail_and_range_support(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    placeholder = import_root / "placeholder.jpg"
    Image.new("RGB", (640, 360), "#a9a69d").save(placeholder)
    thumbnail_source = BytesIO()
    Image.new("RGB", (1280, 720), "#e05b3f").save(thumbnail_source, format="JPEG")
    settings = Settings(import_root=import_root)
    storage = ObjectStorage(settings)
    storage.download_uri = AsyncMock(return_value=thumbnail_source.getvalue())  # type: ignore[method-assign]
    storage.download_uri_response = AsyncMock(  # type: ignore[method-assign]
        return_value=ObjectDownload(
            content=b"video-range",
            content_type="video/mp4",
            content_range="bytes 0-10/1024",
            etag='"video-etag"',
        )
    )
    app = create_app(
        settings=settings,
        asset_repository=FakeVideoAssetRepository(placeholder),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        app.state.object_storage = storage
        thumbnail = client.get(
            "/api/v1/assets/asset_video/thumbnail",
            params={"workspace_id": "workspace_test"},
        )
        content = client.get(
            "/api/v1/assets/asset_video/content",
            params={"workspace_id": "workspace_test"},
            headers={"Range": "bytes=0-10"},
        )

    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/jpeg"
    assert thumbnail.headers["cache-control"] == "public, max-age=86400, immutable"
    with Image.open(BytesIO(thumbnail.content)) as rendered:
        assert max(rendered.size) == 480
    assert content.status_code == 206
    assert content.content == b"video-range"
    assert content.headers["content-type"] == "video/mp4"
    assert content.headers["content-range"] == "bytes 0-10/1024"
    assert content.headers["accept-ranges"] == "bytes"
    storage.download_uri_response.assert_awaited_once_with(
        "s3://capsule/derived/segment.mp4",
        byte_range="bytes=0-10",
    )


def test_legacy_logical_video_range_has_structured_playback_and_no_dead_thumbnail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    source = import_root / "legacy.mkv"
    source.write_bytes(b"0123456789")
    repository = FakeLogicalVideoAssetRepository(
        source,
        source_mime_type="video/x-matroska",
        locator={"start_time_seconds": 1.5, "end_time_seconds": 4.25},
    )
    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=repository,  # type: ignore[arg-type]
    )
    command: list[str] = []

    class _ByteStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunks = chunks

        async def read(self, _size: int = -1) -> bytes:
            return self._chunks.pop(0) if self._chunks else b""

    class _Process:
        pid = 42

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdout = _ByteStream([b"fragmented-mp4"])
            self.stderr = _ByteStream([b""])

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    async def fake_create(*args: str, **_kwargs: object) -> _Process:
        command.extend(args)
        return _Process()

    monkeypatch.setattr(asset_api, "resolve_video_tool", lambda _tool: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(asset_api.asyncio, "create_subprocess_exec", fake_create)

    with TestClient(app) as client:
        listed = client.get("/api/v1/assets", params={"workspace_id": "workspace_test"})
        content = client.get(
            "/api/v1/assets/asset_video/content",
            params={"workspace_id": "workspace_test"},
            headers={"Range": "bytes=2-5"},
        )
        playback = client.get(
            "/api/v1/assets/asset_video/playback",
            params={"workspace_id": "workspace_test"},
        )

    assert listed.status_code == 200
    item = listed.json()["items"][0]
    assert item["preview_url"] is None
    assert item["playback"] == {
        "mode": "transcoded_stream",
        "url": "/api/v1/assets/asset_video/playback?workspace_id=workspace_test",
        "fallback_url": None,
        "mime_type": "video/mp4",
        "start_ms": 1500,
        "end_ms": 4250,
        "duration_ms": 2750,
        "browser_compatible": True,
    }
    assert content.status_code == 206
    assert content.content == b"2345"
    assert content.headers["content-type"] == "video/x-matroska"
    assert content.headers["content-range"] == "bytes 2-5/10"
    assert content.headers["accept-ranges"] == "bytes"
    assert playback.status_code == 200
    assert playback.headers["content-type"] == "video/mp4"
    assert playback.content == b"fragmented-mp4"
    assert command[command.index("-ss") + 1] == "1.500"
    assert command[command.index("-t") + 1] == "2.750"
    assert command[command.index("-i") + 1] == str(source)
    assert command[command.index("-c:v") + 1] in {"h264_videotoolbox", "libx264"}
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[-1] == "pipe:1"


def test_derived_video_clip_playback_keeps_current_interval_metadata(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    source = import_root / "source.mov"
    source.write_bytes(b"source")
    repository = FakeLogicalVideoAssetRepository(
        source,
        source_mime_type="video/quicktime",
        locator={"start_ms": 500, "end_ms": 1_700},
        derived_file_uri="s3://capsule/derived/segment.mp4",
        preview_uri="s3://capsule/derived/preview.jpg",
    )
    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=repository,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        item = client.get("/api/v1/assets", params={"workspace_id": "workspace_test"}).json()[
            "items"
        ][0]

    assert item["preview_url"] == "/api/v1/assets/asset_video/thumbnail?workspace_id=workspace_test"
    assert item["playback"] == {
        "mode": "derived_clip",
        "url": "/api/v1/assets/asset_video/content?workspace_id=workspace_test",
        "fallback_url": None,
        "mime_type": "video/mp4",
        "start_ms": 500,
        "end_ms": 1700,
        "duration_ms": 1200,
        "browser_compatible": True,
    }


def test_h265_mp4_source_range_includes_trusted_transcode_fallback(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    source = import_root / "h265-source.mp4"
    source.write_bytes(b"h265-container")
    repository = FakeLogicalVideoAssetRepository(
        source,
        source_mime_type="video/mp4",
        locator={"start_ms": 250, "end_ms": 1_250, "codec_name": "hevc"},
    )
    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=repository,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        item = client.get("/api/v1/assets", params={"workspace_id": "workspace_test"}).json()[
            "items"
        ][0]

    assert item["playback"] == {
        "mode": "source_range",
        "url": "/api/v1/assets/asset_video/content?workspace_id=workspace_test",
        "fallback_url": "/api/v1/assets/asset_video/playback?workspace_id=workspace_test",
        "mime_type": "video/mp4",
        "start_ms": 250,
        "end_ms": 1250,
        "duration_ms": 1000,
        "browser_compatible": True,
    }


async def test_interrupted_transcode_escalates_from_term_to_kill(monkeypatch) -> None:
    signals: list[int] = []

    class _HangingProcess:
        pid = 43
        returncode: int | None = None

        def __init__(self) -> None:
            self.wait_calls = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                await asyncio.Event().wait()
            self.returncode = -signal.SIGKILL
            return self.returncode

    process = _HangingProcess()
    monkeypatch.setattr(asset_api.os, "getpgid", lambda _pid: 43)
    monkeypatch.setattr(asset_api.os, "killpg", lambda _pid, value: signals.append(value))
    monkeypatch.setattr(asset_api, "_TRANSCODE_GRACE_SECONDS", 0.001)

    await asset_api._terminate_transcode_process(process)  # type: ignore[arg-type]

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.returncode == -signal.SIGKILL


def test_video_source_root_allows_external_source_but_unlisted_path_is_forbidden(
    tmp_path: Path,
) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    external_root = tmp_path / "trusted-videos"
    external_root.mkdir()
    source = external_root / "outside-import.mkv"
    source.write_bytes(b"external-video")
    repository = FakeLogicalVideoAssetRepository(
        source,
        source_mime_type="video/x-matroska",
        locator={"start_ms": 0, "end_ms": 1_000},
    )
    allowed_app = create_app(
        settings=Settings(import_root=import_root, video_source_roots=[external_root]),
        asset_repository=repository,  # type: ignore[arg-type]
    )
    denied_app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=repository,  # type: ignore[arg-type]
    )

    with TestClient(allowed_app) as client:
        allowed = client.get(
            "/api/v1/assets/asset_video/content",
            params={"workspace_id": "workspace_test"},
        )
    with TestClient(denied_app) as client:
        denied = client.get(
            "/api/v1/assets/asset_video/content",
            params={"workspace_id": "workspace_test"},
        )

    assert allowed.status_code == 200
    assert allowed.content == b"external-video"
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "asset_media_outside_allowed_roots"


def test_remote_incompatible_source_refuses_transcode_without_downloading(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    placeholder = import_root / "placeholder.mkv"
    placeholder.write_bytes(b"not-remote-content")

    class _RemoteSourceRepository(FakeLogicalVideoAssetRepository):
        async def list_asset_views(self, **_: object) -> AssetListResponse:
            remote = self.asset.model_copy(
                update={"source_storage_uri": "s3://capsule/source/legacy.mkv"}
            )
            return AssetListResponse(items=[remote], total=1, limit=100, offset=0)

        async def get_asset_view(self, **_: object) -> AssetViewRecord:
            return self.asset.model_copy(
                update={"source_storage_uri": "s3://capsule/source/legacy.mkv"}
            )

        async def get_asset_media(self, **_: object) -> AssetMediaTarget:
            target = await super().get_asset_media()
            return AssetMediaTarget(
                asset_id=target.asset_id,
                workspace_id=target.workspace_id,
                asset_type=target.asset_type,
                source_storage_uri="s3://capsule/source/legacy.mkv",
                source_mime_type=target.source_mime_type,
                preview_uri=None,
                derived_file_uri=None,
            )

    app = create_app(
        settings=Settings(import_root=import_root),
        asset_repository=_RemoteSourceRepository(
            placeholder,
            source_mime_type="video/x-matroska",
            locator={"start_ms": 0, "end_ms": 1_000},
        ),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        listed = client.get(
            "/api/v1/assets",
            params={"workspace_id": "workspace_test"},
        )
        response = client.get(
            "/api/v1/assets/asset_video/playback",
            params={"workspace_id": "workspace_test"},
        )

    assert listed.json()["items"][0]["playback"] == {
        "mode": "source_range",
        "url": "/api/v1/assets/asset_video/content?workspace_id=workspace_test",
        "fallback_url": None,
        "mime_type": "video/x-matroska",
        "start_ms": 0,
        "end_ms": 1000,
        "duration_ms": 1000,
        "browser_compatible": False,
    }
    assert response.status_code == 501
    assert response.json()["detail"]["code"] == "asset_transcode_remote_source_unsupported"
