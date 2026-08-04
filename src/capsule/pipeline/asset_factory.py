"""Build canonical Asset values from parser output."""

import hashlib
import json
from pathlib import Path

from capsule.db.base import id_factory
from capsule.enums import AssetIndexRole
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile


class AssetFactory:
    """Apply source-level fields and stable identities to parsed chunks."""

    def __init__(self) -> None:
        self._new_asset_id = id_factory("asset")

    def build_many(
        self,
        *,
        workspace_id: str,
        source_file_id: str,
        source_sha256: str,
        source_file: DiscoveredFile,
        drafts: list[AssetDraft],
        generation: int = 0,
    ) -> list[AssetCreate]:
        hierarchy_targets: dict[str, AssetDraft] = {}
        for draft in drafts:
            if draft.hierarchy_key is None:
                continue
            if draft.hierarchy_key in hierarchy_targets:
                raise ValueError(f"duplicate hierarchy_key: {draft.hierarchy_key}")
            hierarchy_targets[draft.hierarchy_key] = draft

        parent_asset_keys: dict[int, str] = {}
        for position, draft in enumerate(drafts):
            if draft.index_role != AssetIndexRole.CHILD:
                continue
            parent = hierarchy_targets.get(draft.parent_hierarchy_key or "")
            if parent is None:
                raise ValueError(
                    "child asset references an unknown parent_hierarchy_key: "
                    f"{draft.parent_hierarchy_key}"
                )
            if parent.index_role != AssetIndexRole.PARENT:
                raise ValueError("child asset parent_hierarchy_key must reference a parent asset")
            parent_asset_keys[position] = _asset_key(parent)

        return [
            self.build(
                workspace_id=workspace_id,
                source_file_id=source_file_id,
                source_sha256=source_sha256,
                source_file=source_file,
                draft=draft,
                generation=generation,
                parent_asset_key=parent_asset_keys.get(position),
            )
            for position, draft in enumerate(drafts)
        ]

    def build(
        self,
        *,
        workspace_id: str,
        source_file_id: str,
        source_sha256: str,
        source_file: DiscoveredFile,
        draft: AssetDraft,
        generation: int = 0,
        parent_asset_key: str | None = None,
    ) -> AssetCreate:
        locator_json = _canonical_json(draft.source_locator)
        asset_key = _asset_key(draft)
        content_hash = _sha256_text(
            "\n".join(
                (
                    source_sha256,
                    draft.asset_type.value,
                    locator_json,
                    draft.raw_content or "",
                    _canonical_json(draft.file_info),
                    _canonical_json(
                        [context.model_dump(mode="json") for context in draft.source_contexts]
                    ),
                )
            )
        )
        return AssetCreate(
            asset_id=self._new_asset_id(),
            workspace_id=workspace_id,
            source_file_id=source_file_id,
            asset_type=draft.asset_type,
            file_name=Path(source_file.path).name,
            file_type=source_file.extension,
            asset_key=asset_key,
            index_role=draft.index_role,
            child_order=draft.child_order,
            parent_asset_key=parent_asset_key,
            generation=generation,
            content_hash=content_hash,
            file_tree_context=_file_tree_context(source_file.relative_path),
            source_contexts=draft.source_contexts,
            file_info=draft.file_info,
            source_locator=draft.source_locator,
            raw_content=draft.raw_content,
            derived_file_uri=draft.derived_file_uri,
            preview_uri=draft.preview_uri,
            transient_keyframe_jpegs=draft.transient_keyframe_jpegs,
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _asset_key(draft: AssetDraft) -> str:
    return _sha256_text(f"{draft.asset_type.value}\n{_canonical_json(draft.source_locator)}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_tree_context(relative_path: str) -> list[str]:
    parent = Path(relative_path).parent
    return [] if parent == Path(".") else list(parent.parts)
