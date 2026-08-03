from pathlib import Path

from capsule.enums import AssetType
from capsule.pipeline.asset_factory import AssetFactory
from capsule.schemas import AssetDraft, DiscoveredFile


def test_factory_builds_complete_markdown_asset(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# 标题\n\n正文", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path="docs/notes.md",
        extension=".md",
        size_bytes=path.stat().st_size,
    )

    asset = AssetFactory().build(
        workspace_id="workspace_demo",
        source_file_id="src_01J00000000000000000000000",
        source_sha256="a" * 64,
        source_file=source,
        draft=AssetDraft(
            asset_type=AssetType.MARKDOWN_BLOCK,
            file_name="ignored-derived-name.md",
            source_locator={"block_index": 0, "char_start": 0, "char_end": 8},
            raw_content="# 标题\n\n正文",
        ),
    )

    assert asset.asset_id.startswith("asset_")
    assert len(asset.asset_id.removeprefix("asset_")) == 26
    assert asset.file_name == "notes.md"
    assert asset.file_type == ".md"
    assert asset.file_tree_context == ["docs"]
    assert asset.raw_content == "# 标题\n\n正文"
    assert len(asset.asset_key) == 64
    assert len(asset.content_hash) == 64


def test_asset_key_is_stable_but_content_hash_tracks_content(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("text", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path="notes.md",
        extension=".md",
        size_bytes=4,
    )
    factory = AssetFactory()
    common = {
        "workspace_id": "workspace_demo",
        "source_file_id": "src_01J00000000000000000000000",
        "source_sha256": "a" * 64,
        "source_file": source,
    }
    first = factory.build(
        **common,
        draft=AssetDraft(
            asset_type=AssetType.MARKDOWN_BLOCK,
            file_name="notes.md",
            source_locator={"block_index": 0},
            raw_content="first",
        ),
    )
    second = factory.build(
        **common,
        draft=AssetDraft(
            asset_type=AssetType.MARKDOWN_BLOCK,
            file_name="notes.md",
            source_locator={"block_index": 0},
            raw_content="second",
        ),
    )

    assert first.asset_key == second.asset_key
    assert first.content_hash != second.content_hash
    assert first.file_tree_context == []


def test_transient_keyframes_are_forwarded_without_affecting_hash_or_serialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "movie.mp4"
    path.write_bytes(b"video")
    source = DiscoveredFile(
        path=str(path),
        relative_path="movie.mp4",
        extension=".mp4",
        size_bytes=path.stat().st_size,
    )
    common = {
        "workspace_id": "workspace_demo",
        "source_file_id": "src_01J00000000000000000000000",
        "source_sha256": "a" * 64,
        "source_file": source,
    }
    factory = AssetFactory()
    first = factory.build(
        **common,
        draft=AssetDraft(
            asset_type=AssetType.VIDEO_SEGMENT,
            file_name=path.name,
            source_locator={"start_ms": 0, "end_ms": 1_000},
            transient_keyframe_jpegs=[b"first-cache"],
        ),
    )
    second = factory.build(
        **common,
        draft=AssetDraft(
            asset_type=AssetType.VIDEO_SEGMENT,
            file_name=path.name,
            source_locator={"start_ms": 0, "end_ms": 1_000},
            transient_keyframe_jpegs=[b"second-cache"],
        ),
    )

    assert first.transient_keyframe_jpegs == [b"first-cache"]
    assert first.content_hash == second.content_hash
    assert "transient_keyframe_jpegs" not in first.model_dump(mode="json")
