import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any

from capsule.enums import AssetType, EmbeddingType
from capsule.search.models import (
    ChannelMatch,
    FusedHit,
    MatchedChannel,
    SearchAssetRecord,
    SearchResult,
    SourceFileResult,
)

logger = logging.getLogger(__name__)


class SearchResultBuilder:
    def __init__(self, *, same_source_limit: int = 3) -> None:
        if same_source_limit < 1:
            raise ValueError("same_source_limit must be positive")
        self._same_source_limit = same_source_limit

    @staticmethod
    def validate_hits(
        *,
        ranked_hits: list[FusedHit],
        assets: Mapping[str, SearchAssetRecord],
        workspace_id: str,
        sort_by_score: bool = False,
    ) -> list[FusedHit]:
        """Keep only vectors whose PostgreSQL state and Asset revision are current."""

        validated: list[FusedHit] = []
        for hit in ranked_hits:
            asset = assets.get(hit.asset_id)
            if asset is None or asset.workspace_id != workspace_id:
                continue
            current_by_channel: dict[str, ChannelMatch] = {}
            for match in sorted(hit.matched_channels, key=lambda item: item.rank):
                if (
                    match.embedding_id in asset.indexed_embedding_ids
                    and match.embedding_revision == asset.embedding_revision
                ):
                    current_by_channel.setdefault(match.channel, match)
            current_channels = list(current_by_channel.values())
            if not current_channels:
                logger.info(
                    "Milvus candidate has no current indexed PostgreSQL embedding",
                    extra={"asset_id": hit.asset_id, "workspace_id": workspace_id},
                )
                continue
            validated.append(
                FusedHit(
                    asset_id=asset.asset_id,
                    source_file_id=asset.source_file_id,
                    asset_type=asset.asset_type,
                    score=sum(item.fusion_contribution for item in current_channels),
                    matched_channels=current_channels,
                )
            )
        if sort_by_score:
            return sorted(validated, key=lambda item: (-item.score, item.asset_id))
        return validated

    def build(
        self,
        *,
        ranked_hits: list[FusedHit],
        assets: Mapping[str, SearchAssetRecord],
        workspace_id: str,
        allowed_asset_types: tuple[str, ...],
        top_k: int,
        rerank_items: Mapping[str, tuple[float, str]] | None = None,
    ) -> list[SearchResult]:
        candidates: list[SearchResult] = []
        rerank_items = rerank_items or {}
        ranked_hits = self.validate_hits(
            ranked_hits=ranked_hits,
            assets=assets,
            workspace_id=workspace_id,
        )
        for hit in ranked_hits:
            asset = assets.get(hit.asset_id)
            if asset is None:
                logger.info(
                    "Milvus candidate was rejected by PostgreSQL hydration or filters",
                    extra={"asset_id": hit.asset_id, "workspace_id": workspace_id},
                )
                continue
            if asset.workspace_id != workspace_id:
                logger.error(
                    "cross-workspace search result rejected",
                    extra={"asset_id": hit.asset_id, "workspace_id": workspace_id},
                )
                continue
            if allowed_asset_types and asset.asset_type not in allowed_asset_types:
                logger.error(
                    "search result violates the asset_type filter",
                    extra={"asset_id": hit.asset_id, "asset_type": asset.asset_type},
                )
                continue
            try:
                asset_type = AssetType(asset.asset_type)
            except ValueError:
                logger.error(
                    "search result has unsupported asset type",
                    extra={"asset_id": asset.asset_id, "asset_type": asset.asset_type},
                )
                continue
            ordered_channels = sorted(
                hit.matched_channels,
                key=lambda item: (-item.fusion_contribution, item.rank, item.channel),
            )
            rerank_score, reason = rerank_items.get(asset.asset_id, (None, None))
            candidates.append(
                SearchResult(
                    asset_id=asset.asset_id,
                    asset_type=asset_type,
                    workspace_id=asset.workspace_id,
                    project_id=asset.project_id,
                    source_file_id=asset.source_file_id,
                    file_name=asset.file_name,
                    file_type=asset.file_type,
                    asset_key=asset.asset_key,
                    content_hash=asset.content_hash,
                    asset_name=asset.asset_name,
                    asset_name_source=asset.asset_name_source,
                    asset_description=asset.asset_description,
                    asset_features=asset.asset_features,
                    file_tree_context=asset.file_tree_context,
                    source_contexts=asset.source_contexts,
                    file_info=asset.file_info,
                    source_locator=dict(asset.source_locator),
                    raw_content=asset.raw_content,
                    derived_file_uri=asset.derived_file_uri,
                    preview_uri=asset.preview_uri,
                    processing_status=asset.processing_status,
                    feature_revision=asset.feature_revision,
                    embedding_revision=asset.embedding_revision,
                    error_message=asset.error_message,
                    created_at=asset.created_at,
                    updated_at=asset.updated_at,
                    source_file=SourceFileResult(
                        source_file_id=asset.source_file_id,
                        workspace_id=asset.source_workspace_id,
                        project_id=asset.source_project_id,
                        original_file_name=asset.source_file_name,
                        file_type=asset.source_file_type,
                        mime_type=asset.source_mime_type,
                        relative_path=asset.source_relative_path,
                        file_tree_context=asset.source_file_tree_context,
                        storage_uri=asset.source_storage_uri,
                        sha256=asset.source_sha256,
                        file_size_bytes=asset.source_file_size_bytes,
                        processing_status=asset.source_processing_status,
                        error_message=asset.source_error_message,
                        created_at=asset.source_created_at,
                        updated_at=asset.source_updated_at,
                    ),
                    score=hit.score,
                    matched_channels=[
                        MatchedChannel(
                            channel=match.channel,
                            embedding_type=match.embedding_type,
                            embedding_id=match.embedding_id,
                            embedding_revision=match.embedding_revision,
                            rank=match.rank,
                            similarity=match.similarity,
                            fusion_contribution=match.fusion_contribution,
                            rrf_contribution=match.rrf_contribution,
                        )
                        for match in ordered_channels
                    ],
                    matched_feature=_matched_feature(
                        asset.asset_features,
                        ordered_channels,
                    ),
                    matched_reason=reason or _default_reason(ordered_channels),
                    rerank_score=rerank_score,
                    folded_asset_ids=[asset.asset_id],
                )
            )
        return _fold_and_limit(
            candidates,
            same_source_limit=self._same_source_limit,
            top_k=top_k,
        )


def _fold_and_limit(
    candidates: list[SearchResult],
    *,
    same_source_limit: int,
    top_k: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    source_counts: Counter[str] = Counter()
    for candidate in candidates:
        source_id = (
            candidate.source_file.source_file_id if candidate.source_file is not None else ""
        )
        folded_into = next(
            (
                existing
                for existing in results
                if existing.source_file is not None
                and existing.source_file.source_file_id == source_id
                and _can_fold(existing, candidate)
            ),
            None,
        )
        if folded_into is not None:
            folded_into.folded_asset_ids.extend(candidate.folded_asset_ids)
            folded_into.group_kind = (
                "video_segments"
                if candidate.asset_type is AssetType.VIDEO_SEGMENT
                else "markdown_blocks"
            )
            folded_into.source_contexts = _merge_contexts(
                folded_into.source_contexts,
                candidate.source_contexts,
            )
            _expand_locator(folded_into.source_locator, candidate.source_locator)
            continue
        if source_counts[source_id] >= same_source_limit:
            continue
        source_counts[source_id] += 1
        results.append(candidate)
        if len(results) >= top_k:
            break
    return results


def _can_fold(left: SearchResult, right: SearchResult) -> bool:
    if left.asset_type is not right.asset_type:
        return False
    if left.asset_type is AssetType.MARKDOWN_BLOCK:
        left_index = _number(left.source_locator, "block_index")
        right_index = _number(right.source_locator, "block_index")
        return (
            left_index is not None
            and right_index is not None
            and abs(left_index - right_index) <= 1
        )
    if left.asset_type is not AssetType.VIDEO_SEGMENT:
        return False
    left_start = _first_number(
        left.source_locator,
        "start_ms",
        "start_time_ms",
        "start_seconds",
        "start_time_seconds",
    )
    left_end = _first_number(
        left.source_locator,
        "end_ms",
        "end_time_ms",
        "end_seconds",
        "end_time_seconds",
    )
    right_start = _first_number(
        right.source_locator,
        "start_ms",
        "start_time_ms",
        "start_seconds",
        "start_time_seconds",
    )
    right_end = _first_number(
        right.source_locator,
        "end_ms",
        "end_time_ms",
        "end_seconds",
        "end_time_seconds",
    )
    if None in (left_start, left_end, right_start, right_end):
        return False
    assert left_start is not None and left_end is not None
    assert right_start is not None and right_end is not None
    uses_milliseconds = any(
        key in left.source_locator or key in right.source_locator
        for key in ("start_ms", "end_ms", "start_time_ms", "end_time_ms")
    )
    gap_limit = 2_000.0 if uses_milliseconds else 2.0
    gap = max(right_start - left_end, left_start - right_end, 0.0)
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shortest_duration = min(left_end - left_start, right_end - right_start)
    overlap_ratio = overlap / shortest_duration if shortest_duration > 0 else 0.0
    return gap < gap_limit or overlap_ratio > 0.30


def _expand_locator(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for start_key, end_key in (
        ("start_ms", "end_ms"),
        ("start_time_ms", "end_time_ms"),
        ("start_seconds", "end_seconds"),
        ("start_time_seconds", "end_time_seconds"),
        ("block_index", "block_index"),
    ):
        left_start = _number(target, start_key)
        right_start = _number(incoming, start_key)
        left_end = _number(target, end_key)
        right_end = _number(incoming, end_key)
        if left_start is not None and right_start is not None:
            target[start_key] = min(left_start, right_start)
        if left_end is not None and right_end is not None:
            target[end_key] = max(left_end, right_end)


def _number(locator: Mapping[str, Any], key: str) -> float | None:
    value = locator.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _first_number(locator: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(locator, key)
        if value is not None:
            return value
    return None


def _merge_contexts(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(left)
    seen = {repr(item) for item in merged}
    for item in right:
        if repr(item) not in seen:
            merged.append(item)
            seen.add(repr(item))
    return merged


def _matched_feature(
    features: Mapping[str, Any],
    channels: list[ChannelMatch],
) -> str | None:
    feature_types = {
        item: item.value
        for item in EmbeddingType
        if item
        not in {
            EmbeddingType.NATIVE_MULTIMODAL,
            EmbeddingType.ASSET_DESCRIPTION,
        }
    }
    for channel in channels:
        key = feature_types.get(channel.embedding_type)
        if key is None:
            continue
        raw = features.get(key)
        if isinstance(raw, str) and raw:
            return raw
        if isinstance(raw, Mapping):
            for value_key in ("effective_value", "value", "user_value", "model_value"):
                value = raw.get(value_key)
                if isinstance(value, str) and value:
                    return value
    return None


def _default_reason(channels: list[ChannelMatch]) -> str | None:
    if not channels:
        return None
    names = "、".join(item.embedding_type.value for item in channels[:3])
    return f"命中检索维度：{names}"
