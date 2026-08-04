from pathlib import Path

import pytest
from sqlalchemy import UniqueConstraint

from capsule.db.models import Asset
from capsule.enums import AssetIndexRole, AssetType
from capsule.pipeline.asset_factory import AssetFactory
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile


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


def test_build_many_resolves_parent_hierarchy_to_stable_asset_key(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("text", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path="notes.md",
        extension=".md",
        size_bytes=4,
    )

    assets = AssetFactory().build_many(
        workspace_id="workspace_demo",
        source_file_id="src_01J00000000000000000000000",
        source_sha256="a" * 64,
        source_file=source,
        drafts=[
            AssetDraft(
                asset_type=AssetType.MARKDOWN_BLOCK,
                file_name=path.name,
                index_role=AssetIndexRole.PARENT,
                hierarchy_key="section:one",
                source_locator={"block_index": 0},
                raw_content="Parent summary",
            ),
            AssetDraft(
                asset_type=AssetType.MARKDOWN_BLOCK,
                file_name=path.name,
                parent_hierarchy_key="section:one",
                child_order=0,
                source_locator={"block_index": 1},
                raw_content="Child content",
            ),
        ],
    )

    parent, child = assets
    assert parent.index_role == AssetIndexRole.PARENT
    assert parent.parent_asset_key is None
    assert child.index_role == AssetIndexRole.CHILD
    assert child.parent_asset_key == parent.asset_key
    assert child.child_order == 0
    assert "parent_asset_key" not in child.model_dump(mode="json")


def test_build_many_rejects_unknown_or_non_parent_hierarchy_target(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("text", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path="notes.md",
        extension=".md",
        size_bytes=4,
    )
    common = {
        "workspace_id": "workspace_demo",
        "source_file_id": "src_01J00000000000000000000000",
        "source_sha256": "a" * 64,
        "source_file": source,
    }

    with pytest.raises(ValueError, match="unknown parent_hierarchy_key"):
        AssetFactory().build_many(
            **common,
            drafts=[
                AssetDraft(
                    asset_type=AssetType.MARKDOWN_BLOCK,
                    file_name=path.name,
                    parent_hierarchy_key="missing",
                    child_order=0,
                    source_locator={"block_index": 0},
                )
            ],
        )


def test_parent_child_order_uniqueness_is_deferred_for_reprocessing() -> None:
    """A regenerated child may temporarily share its stale predecessor's order."""
    constraint = next(
        constraint
        for constraint in Asset.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_asset_parent_child_order"
    )

    assert constraint.deferrable is True
    assert constraint.initially == "DEFERRED"


def test_asset_create_rejects_child_without_parent_asset_key() -> None:
    with pytest.raises(ValueError, match="parent_asset_key"):
        AssetCreate(
            asset_id="asset_child",
            workspace_id="workspace_demo",
            source_file_id="src_demo",
            asset_type=AssetType.TEXT_BLOCK,
            file_name="notes.txt",
            file_type=".txt",
            asset_key="child-key",
            index_role=AssetIndexRole.CHILD,
            child_order=0,
            content_hash="a" * 64,
        )


def test_public_build_rejects_unresolved_child(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("child", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=".txt",
        size_bytes=path.stat().st_size,
    )

    with pytest.raises(ValueError, match="parent_asset_key"):
        AssetFactory().build(
            workspace_id="workspace_demo",
            source_file_id="src_demo",
            source_sha256="a" * 64,
            source_file=source,
            draft=AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                parent_hierarchy_key="parent",
                child_order=0,
                source_locator={"block_index": 1},
            ),
        )


def test_hierarchy_changes_do_not_change_asset_identity_or_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("same content", encoding="utf-8")
    source = DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=".txt",
        size_bytes=path.stat().st_size,
    )
    common = {
        "workspace_id": "workspace_demo",
        "source_file_id": "src_demo",
        "source_sha256": "a" * 64,
        "source_file": source,
    }
    first = AssetFactory().build_many(
        **common,
        drafts=[
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                index_role=AssetIndexRole.PARENT,
                hierarchy_key="parent:first",
                source_locator={"block_index": 0},
            ),
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                parent_hierarchy_key="parent:first",
                child_order=0,
                source_locator={"block_index": 1},
                raw_content="same content",
            ),
        ],
    )[1]
    second = AssetFactory().build_many(
        **common,
        drafts=[
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                index_role=AssetIndexRole.PARENT,
                hierarchy_key="parent:second",
                source_locator={"block_index": 2},
            ),
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                parent_hierarchy_key="parent:second",
                child_order=7,
                source_locator={"block_index": 1},
                raw_content="same content",
            ),
        ],
    )[1]

    assert first.asset_key == second.asset_key
    assert first.content_hash == second.content_hash
    assert first.parent_asset_key != second.parent_asset_key
    assert first.child_order != second.child_order
