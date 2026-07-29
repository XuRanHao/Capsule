from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.db.repositories import AssetMediaTarget
from capsule.enums import AssetType
from capsule.schemas import AssetListResponse, AssetSourceRecord, AssetViewRecord


class FakeAssetRepository:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
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
            derived_file_uri=None,
        )


def test_asset_list_and_local_preview_are_available(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    import_root.mkdir()
    image_path = import_root / "image.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
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
        assert payload["items"][0]["preview_url"].endswith(
            "/api/v1/assets/asset_image/preview?workspace_id=workspace_test"
        )

        preview = client.get(
            "/api/v1/assets/asset_image/preview",
            params={"workspace_id": "workspace_test"},
        )
        assert preview.status_code == 200
        assert preview.content == image_path.read_bytes()
