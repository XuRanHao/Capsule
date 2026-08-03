"""Build canonical Asset values from parser output."""

import hashlib
import json
from pathlib import Path

from capsule.db.base import id_factory
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
        return [
            self.build(
                workspace_id=workspace_id,
                source_file_id=source_file_id,
                source_sha256=source_sha256,
                source_file=source_file,
                draft=draft,
                generation=generation,
            )
            for draft in drafts
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
    ) -> AssetCreate:
        locator_json = _canonical_json(draft.source_locator)
        asset_key = _sha256_text(f"{draft.asset_type.value}\n{locator_json}")
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_tree_context(relative_path: str) -> list[str]:
    parent = Path(relative_path).parent
    return [] if parent == Path(".") else list(parent.parts)
