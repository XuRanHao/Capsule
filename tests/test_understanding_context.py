import base64
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

from capsule.config import Settings
from capsule.db.repositories import EmbeddingAsset
from capsule.enums import AssetType
from capsule.pipeline.understanding import (
    _DESCRIPTION_CONTEXT_RULES,
    _SUBJECT_OUTPUT_RULES,
    AssetUnderstandingService,
    _asset_context_payload,
    _content_context_payload,
)


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
    assert payload["entity_hints"] == [
        {"source": "directory", "value": "images", "scope": "collection"},
        {"source": "document_title", "value": "光线参考", "scope": "document"},
        {"source": "heading", "value": "午后黄昏", "scope": "section"},
        {"source": "file_name", "value": "sunset", "scope": "asset"},
    ]
    assert "source_uri" not in str(payload)
    assert "asset_storage_uri" not in str(payload)
    assert "source_storage_uri" not in str(payload)

    content_payload = _content_context_payload(asset)
    assert content_payload == {
        "asset_type": "image",
        "file_info": {},
    }
    assert "sunset" not in str(content_payload)
    assert "images" not in str(content_payload)


def test_description_and_subject_rules_use_positive_independent_paths() -> None:
    assert "描述直接呈现可观察或可阅读的事实" in _DESCRIPTION_CONTEXT_RULES
    assert "元数据由独立路径处理" in _DESCRIPTION_CONTEXT_RULES
    assert "无需在内容描述中解释其来源" in _DESCRIPTION_CONTEXT_RULES
    assert "必须" not in _DESCRIPTION_CONTEXT_RULES
    assert "subject_content 走内容提取路径" in _SUBJECT_OUTPUT_RULES
    assert "元数据实体会由独立路径提取" in _SUBJECT_OUTPUT_RULES
    assert "必须" not in _SUBJECT_OUTPUT_RULES
    assert "salience 使用 0 到 1 的相对数值" in _SUBJECT_OUTPUT_RULES


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

    assert "主体维度把不同实体拆成不同 item" in messages[0]["content"]
    assert "subject 写简短稳定的实体名称" in messages[0]["content"]
    assert "元数据实体会由独立路径提取" in messages[0]["content"]
    assert "三个互补的检索视角" in messages[0]["content"]
    assert "分别完成三次聚焦" in messages[0]["content"]
    assert "不要把“冷蓝逆光”作为主体特征" in messages[0]["content"]
    assert "根据当前维度相关信息在素材中的占比、显著性和丰富程度" in messages[0]["content"]
    assert "scene_theme 在素材具有可辨识" not in messages[0]["content"]
    assert "mood_atmosphere 依据画面或文本" not in messages[0]["content"]
    assert "人物内心、动机或性格只有在素材明确呈现时" not in messages[0]["content"]
    assert "description 表示当前维度的事实本身" not in messages[0]["content"]

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

    assert image_urls[0].startswith("data:image/jpeg;base64,")
    encoded = image_urls[0].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as prepared:
        assert prepared.size == (768, 768)
