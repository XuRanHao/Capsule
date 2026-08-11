from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from capsule.config import Settings
from capsule.db.repositories import EmbeddingAsset
from capsule.enums import AssetType, FeatureStatus
from capsule.pipeline.understanding import (
    _DESCRIPTION_CONTEXT_RULES,
    AssetUnderstandingService,
    _asset_context_payload,
    _attach_asset_usage_path_context,
    _usage_hint_from_path,
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


def test_generic_storage_path_does_not_create_usage_semantics() -> None:
    assert (
        _usage_hint_from_path(
            source_path="测试素材2（打乱素材集合）/黄.png",
            file_tree_context=["测试素材2（打乱素材集合）"],
        )
        is None
    )


async def test_video_understanding_uses_keyframe_data_uris() -> None:
    class Reader:
        def __init__(self) -> None:
            self.uris: list[str] = []

        async def download_uri(self, uri: str) -> bytes:
            self.uris.append(uri)
            return uri.rsplit("/", 1)[-1].encode()

    asset = EmbeddingAsset(
        asset_id="asset_video",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="source_video",
        asset_type=AssetType.VIDEO_SEGMENT.value,
        file_type=".mp4",
        content_hash="c" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri="s3://capsule/video/segment.mp4",
        source_storage_uri="file:///source.mp4",
        source_mime_type="video/mp4",
        file_info={
            "keyframes": [
                {"uri": "s3://capsule/video/keyframes/01.jpg"},
                {"uri": "s3://capsule/video/keyframes/02.jpg"},
            ]
        },
    )
    reader = Reader()
    service = AssetUnderstandingService(
        settings=Settings(),
        embedding_repository=None,  # type: ignore[arg-type]
        asset_repository=None,  # type: ignore[arg-type]
        model_client=None,  # type: ignore[arg-type]
        artifact_reader=reader,
    )

    messages = await service._messages(asset)
    content = messages[1]["content"]
    image_urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]

    assert "0 到 5 条最具表现力和区分度" in messages[0]["content"]
    assert "每个 Feature 围绕自己的正向语义范围组织事实" in messages[0]["content"]
    assert "整幅内容可辨识的叙事语境" in messages[0]["content"]
    assert "角色三视图、产品白底陈列或孤立元素展示" in messages[0]["content"]
    assert "缺少整体场景语境的主体陈列或孤立元素不适用" in messages[0]["content"]
    assert "可核验的光线、色彩、空间、天气、动作、声音和叙事表现" in messages[0]["content"]
    assert "人物内心、动机或性格只有在素材明确呈现时" in messages[0]["content"]
    assert "来源平台、数据集或采集渠道" in messages[0]["content"]
    assert "桌子 红色；星空 深蓝" in messages[0]["content"]

    assert reader.uris == [
        "s3://capsule/video/keyframes/01.jpg",
        "s3://capsule/video/keyframes/02.jpg",
    ]
    assert image_urls == [
        "data:image/jpeg;base64,MDEuanBn",
        "data:image/jpeg;base64,MDIuanBn",
    ]


async def test_logical_video_understanding_extracts_frames_without_persistent_files() -> None:
    class FrameExtractor:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def extract(self, request) -> list[bytes]:
            self.requests.append(request)
            return [b"\xff\xd8frame-one", b"\xff\xd8frame-two"]

    asset = EmbeddingAsset(
        asset_id="asset_logical_video",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="source_video",
        asset_type=AssetType.VIDEO_SEGMENT.value,
        file_type=".mp4",
        content_hash="e" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri="file:///library/original.mp4",
        source_mime_type="video/mp4",
        file_info={
            "video_output_mode": "logical",
            "representative_frames": [
                {"timestamp_ms": 12_000},
                {"timestamp_ms": 16_000},
            ],
        },
        source_locator={"start_ms": 10_000, "end_ms": 20_000},
    )
    extractor = FrameExtractor()
    service = AssetUnderstandingService(
        settings=Settings(),
        embedding_repository=None,  # type: ignore[arg-type]
        asset_repository=None,  # type: ignore[arg-type]
        model_client=None,  # type: ignore[arg-type]
        video_frame_extractor=extractor,
    )

    messages = await service._messages(asset)
    content = messages[1]["content"]
    image_urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]

    assert image_urls == [
        "data:image/jpeg;base64,/9hmcmFtZS1vbmU=",
        "data:image/jpeg;base64,/9hmcmFtZS10d28=",
    ]
    assert len(extractor.requests) == 1
    request = extractor.requests[0]
    assert request.source_uri == "file:///library/original.mp4"
    assert request.timestamps_ms == (12_000, 16_000)


async def test_document_image_understanding_uses_materialised_image(tmp_path: Path) -> None:
    image_path = tmp_path / "extracted.png"
    Image.new("RGB", (20, 20), "green").save(image_path)
    asset = EmbeddingAsset(
        asset_id="asset_document_image",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="source_document",
        asset_type=AssetType.IMAGE.value,
        file_type=".docx",
        content_hash="d" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 30, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=image_path.as_uri(),
        source_storage_uri="file:///source/document.docx",
        source_mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_info={"mime_type": "image/png", "embedded_in_document": True},
    )
    service = AssetUnderstandingService(
        settings=Settings(),
        embedding_repository=None,  # type: ignore[arg-type]
        asset_repository=None,  # type: ignore[arg-type]
        model_client=None,  # type: ignore[arg-type]
    )

    messages = await service._messages(asset)
    content = messages[1]["content"]
    image_urls = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]

    assert image_urls[0].startswith("data:image/png;base64,")
