from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient
from PIL import Image

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
