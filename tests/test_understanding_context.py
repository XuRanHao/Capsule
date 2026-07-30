from datetime import UTC, datetime

from capsule.db.repositories import EmbeddingAsset
from capsule.enums import AssetType, FeatureStatus
from capsule.pipeline.understanding import (
    _DESCRIPTION_CONTEXT_RULES,
    _asset_context_payload,
    _attach_asset_usage_path_context,
)
from capsule.schemas import AssetUnderstanding


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


def test_description_context_rules_require_semantic_fusion_without_path_repetition() -> None:
    assert "有实际语义的信息必须自然融入描述" in _DESCRIPTION_CONTEXT_RULES
    assert "路径或文字与素材内容冲突时，以素材本身为准" in _DESCRIPTION_CONTEXT_RULES
    assert "忽略纯编号、序号、通用词" in _DESCRIPTION_CONTEXT_RULES
    assert "禁止在结果中机械复述文件名、扩展名、目录、路径" in _DESCRIPTION_CONTEXT_RULES


def test_asset_usage_path_is_persisted_as_metadata_evidence() -> None:
    feature_names = [
        "subject_content",
        "scene_theme",
        "visual_style",
        "color_composition",
        "mood_atmosphere",
        "character_state_or_psychology",
        "asset_usage",
        "target_audience",
        "provenance",
        "rights_version_authorship",
    ]
    understanding = AssetUnderstanding.model_validate(
        {
            "asset_name": "测试海报",
            "asset_description": "一张用于测试的视觉海报。",
            "features": {
                name: {
                    "value": None,
                    "status": "unknown",
                    "confidence": 0,
                    "evidence": [],
                }
                for name in feature_names
            },
        }
    )
    asset = EmbeddingAsset(
        asset_id="asset_usage",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="source_usage",
        asset_type=AssetType.IMAGE.value,
        file_type=".png",
        content_hash="b" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri="file:///temporary/import/海报/素材/20251216-143446.png",
        source_mime_type="image/png",
        file_name="20251216-143446.png",
        source_relative_path="海报/素材/20251216-143446.png",
        file_tree_context=["海报", "素材"],
    )

    _attach_asset_usage_path_context(understanding, asset)

    usage = understanding.features.asset_usage
    assert usage.status is FeatureStatus.METADATA
    assert usage.value == "海报制作"
    assert usage.source_path == "海报/素材/20251216-143446.png"
    assert usage.description is not None
    assert "海报/素材/20251216-143446.png" in usage.description
    assert usage.evidence == ["相对文件路径：海报/素材/20251216-143446.png"]
