from datetime import UTC, datetime

from capsule.db.repositories import EmbeddingAsset
from capsule.enums import AssetType
from capsule.pipeline.understanding import _asset_context_payload


def test_asset_context_payload_includes_source_path_and_linked_paragraph() -> None:
    asset = EmbeddingAsset(
        asset_id="asset_image",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="source_image",
        asset_type=AssetType.IMAGE.value,
        file_type=".png",
        content_hash="a" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri="file:///temporary/import/images/sunset.png",
        source_mime_type="image/png",
        file_name="sunset.png",
        source_relative_path="images/sunset.png",
        source_contexts=[
            {
                "text": "午后黄昏呈现金黄色调。",
                "relation_type": "preceding_text",
                "text_block_index": 1,
                "paragraph_id": "board.md#block-1",
                "source_path": "board.md",
                "document_title": "光线参考",
                "heading_path": ["光线参考", "午后黄昏"],
            }
        ],
    )

    payload = _asset_context_payload(asset)

    assert payload["context"]["source_path"] == "images/sunset.png"
    assert payload["context"]["associated_text"] == ["午后黄昏呈现金黄色调。"]
    assert payload["context"]["heading_path"] == ["光线参考", "午后黄昏"]
    assert "source_uri" not in str(payload)
    assert "asset_storage_uri" not in str(payload)
    assert "source_storage_uri" not in str(payload)
