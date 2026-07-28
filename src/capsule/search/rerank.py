import logging
from collections.abc import Mapping

from capsule.search.contracts import SearchUnderstandingClient
from capsule.search.models import FusedHit, SearchAssetRecord, SearchRequest

logger = logging.getLogger(__name__)


class SearchReranker:
    def __init__(
        self,
        client: SearchUnderstandingClient | None,
        *,
        candidate_limit: int = 30,
    ) -> None:
        self._client = client
        self._candidate_limit = candidate_limit

    async def rerank(
        self,
        *,
        request: SearchRequest,
        image_url: str | None,
        ranked_hits: list[FusedHit],
        assets: Mapping[str, SearchAssetRecord],
    ) -> tuple[list[FusedHit], dict[str, tuple[float, str]], tuple[str, ...]]:
        if self._client is None:
            return ranked_hits, {}, ("rerank model is unavailable; fusion order retained",)

        head = [item for item in ranked_hits if item.asset_id in assets][: self._candidate_limit]
        tail_ids = {item.asset_id for item in head}
        tail = [item for item in ranked_hits if item.asset_id not in tail_ids]
        candidates = [_candidate_payload(item, assets[item.asset_id]) for item in head]
        try:
            response = await self._client.rerank_search_results(
                request,
                image_url=image_url,
                candidates=candidates,
            )
        except Exception:
            logger.warning("search rerank failed; fusion order retained", exc_info=True)
            return ranked_hits, {}, ("rerank failed; fusion order retained",)

        by_id = {item.asset_id: item for item in head}
        annotations: dict[str, tuple[float, str]] = {}
        ordered: list[FusedHit] = []
        seen: set[str] = set()
        for item in sorted(
            response.items,
            key=lambda value: (-value.relevance_score, value.asset_id),
        ):
            hit = by_id.get(item.asset_id)
            if hit is None or item.asset_id in seen:
                continue
            ordered.append(hit)
            seen.add(item.asset_id)
            annotations[item.asset_id] = (item.relevance_score, item.reason)
        ordered.extend(item for item in head if item.asset_id not in seen)
        return ordered + tail, annotations, ()


def _candidate_payload(hit: FusedHit, asset: SearchAssetRecord) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "asset_type": asset.asset_type,
        "asset_name": asset.asset_name,
        "asset_description": asset.asset_description,
        "asset_features": asset.asset_features,
        "source_contexts": asset.source_contexts,
        "source_locator": asset.source_locator,
        "fusion_score": hit.score,
        "channel_scores": [
            {
                "embedding_type": item.embedding_type.value,
                "similarity": item.similarity,
                "contribution": item.fusion_contribution,
            }
            for item in hit.matched_channels
        ],
    }
