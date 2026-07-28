import logging
from time import perf_counter

from capsule.config import Settings
from capsule.search.contracts import AssetSearchRepository
from capsule.search.fusion import WeightedReciprocalRankFusion
from capsule.search.models import (
    SearchQueryEcho,
    SearchRequest,
    SearchResponse,
)
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.recall import MultiChannelRecall
from capsule.search.result_builder import SearchResultBuilder

logger = logging.getLogger(__name__)


class SearchUnavailableError(RuntimeError):
    pass


class SearchService:
    def __init__(
        self,
        *,
        query_embedding: QueryEmbeddingService,
        recall: MultiChannelRecall,
        assets: AssetSearchRepository,
        settings: Settings,
    ) -> None:
        self._query_embedding = query_embedding
        self._recall = recall
        self._assets = assets
        self._fusion = WeightedReciprocalRankFusion(
            rrf_k=settings.search_rrf_k,
            candidate_cap=settings.search_candidate_cap,
        )
        self._result_builder = SearchResultBuilder(
            same_source_limit=settings.search_same_source_limit
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = perf_counter()

        embedding_started = perf_counter()
        plan = await self._query_embedding.embed(request)
        embedding_ms = _elapsed_ms(embedding_started)

        recall_started = perf_counter()
        recall = await self._recall.search(
            plan=plan,
            workspace_id=request.workspace_id,
            asset_types=tuple(item.value for item in request.filters.asset_type),
            top_k=request.top_k,
        )
        recall_ms = _elapsed_ms(recall_started)
        if not recall.channels:
            raise SearchUnavailableError("all Milvus recall channels failed")

        fusion_started = perf_counter()
        ranked = self._fusion.fuse(recall.channels)
        fusion_ms = _elapsed_ms(fusion_started)

        hydration_started = perf_counter()
        assets = await self._assets.get_by_ids(
            workspace_id=request.workspace_id,
            asset_ids=[item.asset_id for item in ranked],
        )
        results = self._result_builder.build(
            ranked_hits=ranked,
            assets=assets,
            workspace_id=request.workspace_id,
            allowed_asset_types=tuple(item.value for item in request.filters.asset_type),
            top_k=request.top_k,
        )
        hydration_ms = _elapsed_ms(hydration_started)

        reasons = list(dict.fromkeys(plan.degraded_reasons + recall.degraded_reasons))
        logger.info(
            "search completed workspace_id=%s query_type=%s results=%d "
            "channels=%d embedding_ms=%.2f recall_ms=%.2f fusion_ms=%.2f "
            "hydration_ms=%.2f total_ms=%.2f degraded=%s",
            request.workspace_id,
            request.query_type.value,
            len(results),
            len(recall.channels),
            embedding_ms,
            recall_ms,
            fusion_ms,
            hydration_ms,
            _elapsed_ms(started),
            bool(reasons),
        )
        for channel in recall.channels:
            logger.info(
                "search channel workspace_id=%s channel=%s hits=%d weight=%.3f",
                request.workspace_id,
                channel.query_vector.channel,
                len(channel.hits),
                channel.query_vector.weight,
            )
        return SearchResponse(
            query=SearchQueryEcho(
                query_type=request.query_type,
                query_text=request.query_text,
                query_image_url=request.query_image_url,
            ),
            total=len(results),
            degraded=bool(reasons),
            degraded_reasons=reasons,
            results=results,
        )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
