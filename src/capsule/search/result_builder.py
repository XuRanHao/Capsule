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

    def build(
        self,
        *,
        ranked_hits: list[FusedHit],
        assets: Mapping[str, SearchAssetRecord],
        workspace_id: str,
        allowed_asset_types: tuple[str, ...],
        top_k: int,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        source_counts: Counter[str] = Counter()
        for hit in ranked_hits:
            asset = assets.get(hit.asset_id)
            if asset is None:
                logger.error(
                    "Milvus asset is missing from PostgreSQL",
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
            if source_counts[asset.source_file_id] >= self._same_source_limit:
                continue
            try:
                asset_type = AssetType(asset.asset_type)
            except ValueError:
                logger.error(
                    "search result has unsupported asset type",
                    extra={"asset_id": asset.asset_id, "asset_type": asset.asset_type},
                )
                continue
            source_counts[asset.source_file_id] += 1
            ordered_channels = sorted(
                hit.matched_channels,
                key=lambda item: (-item.rrf_contribution, item.rank, item.channel),
            )
            results.append(
                SearchResult(
                    asset_id=asset.asset_id,
                    asset_type=asset_type,
                    asset_name=asset.asset_name,
                    asset_description=asset.asset_description,
                    asset_features=asset.asset_features,
                    source_contexts=asset.source_contexts,
                    source_locator=asset.source_locator,
                    preview_uri=asset.preview_uri,
                    source_file=SourceFileResult(
                        source_file_id=asset.source_file_id,
                        original_file_name=asset.source_file_name,
                        file_type=asset.source_file_type,
                        relative_path=asset.source_relative_path,
                    ),
                    score=hit.score,
                    matched_channels=[
                        MatchedChannel(
                            channel=match.channel,
                            embedding_type=match.embedding_type,
                            rank=match.rank,
                            similarity=match.similarity,
                            rrf_contribution=match.rrf_contribution,
                        )
                        for match in ordered_channels
                    ],
                    matched_feature=_matched_feature(
                        asset.asset_features,
                        ordered_channels,
                    ),
                )
            )
            if len(results) >= top_k:
                break
        return results


def _matched_feature(
    features: Mapping[str, Any],
    channels: list[ChannelMatch],
) -> str | None:
    feature_types = {
        EmbeddingType.SUBJECT_CONTENT: "subject_content",
        EmbeddingType.VISUAL_STYLE: "visual_style",
    }
    for channel in channels:
        key = feature_types.get(channel.embedding_type)
        if key is None:
            continue
        raw = features.get(key)
        if isinstance(raw, str) and raw:
            return raw
        if isinstance(raw, Mapping):
            value = raw.get("value")
            if isinstance(value, str) and value:
                return value
    return None
