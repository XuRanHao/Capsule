import logging
from time import perf_counter

from capsule.config import Settings
from capsule.search.contracts import AssetSearchRepository, QueryImageResolver
from capsule.search.fusion import FusionEngine
from capsule.search.history import SearchHistoryRepository
from capsule.search.models import (
    RerankMethod,
    SearchQueryEcho,
    SearchRequest,
    SearchResponse,
    SearchTimings,
)
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.rerank import SearchReranker
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
        query_parser: QueryParser | None = None,
        reranker: SearchReranker | None = None,
        history: SearchHistoryRepository | None = None,
        image_resolver: QueryImageResolver | None = None,
    ) -> None:
        self._query_embedding = query_embedding
        self._query_parser = query_parser or QueryParser()
        self._recall = recall
        self._assets = assets
        self._fusion = FusionEngine(
            rrf_k=settings.search_rrf_k,
            candidate_cap=settings.search_candidate_cap,
        )
        self._reranker = reranker or SearchReranker(None)
        self._result_builder = SearchResultBuilder(
            same_source_limit=settings.search_same_source_limit
        )
        self._history = history
        self._image_resolver = image_resolver
        self._settings = settings

    async def search(
        self,
        request: SearchRequest,
        *,
        existing_capsule_id: str | None = None,
    ) -> SearchResponse:
        started = perf_counter()
        reasons: list[str] = []
        image_url = await self._resolve_image(request)

        parser_started = perf_counter()
        parsed_query, parser_reasons = await self._query_parser.parse(
            request,
            image_url=image_url,
        )
        reasons.extend(parser_reasons)
        parser_ms = _elapsed_ms(parser_started)

        embedding_started = perf_counter()
        plan = await self._query_embedding.embed(
            request,
            parsed_query,
            image_url=image_url,
        )
        reasons.extend(plan.degraded_reasons)
        embedding_ms = _elapsed_ms(embedding_started)

        recall_started = perf_counter()
        recall = await self._recall.search(
            plan=plan,
            workspace_id=request.workspace_id,
            filters=request.filters,
            top_k=request.top_k,
        )
        reasons.extend(recall.degraded_reasons)
        recall_ms = _elapsed_ms(recall_started)
        if not recall.channels:
            raise SearchUnavailableError("all Milvus recall channels failed")

        fusion_started = perf_counter()
        ranked = self._fusion.fuse(recall.channels, request.fusion_method)
        fusion_ms = _elapsed_ms(fusion_started)

        hydration_started = perf_counter()
        assets = await self._assets.get_by_ids(
            workspace_id=request.workspace_id,
            asset_ids=[item.asset_id for item in ranked],
            embedding_ids=[
                match.embedding_id
                for item in ranked
                for match in item.matched_channels
            ],
            created_by=request.created_by,
            filters=request.filters,
        )
        ranked = self._result_builder.validate_hits(
            ranked_hits=ranked,
            assets=assets,
            workspace_id=request.workspace_id,
            sort_by_score=True,
        )
        hydration_ms = _elapsed_ms(hydration_started)

        rerank_started = perf_counter()
        rerank_items: dict[str, tuple[float, str]] = {}
        if request.rerank_method is RerankMethod.DOUBAO_SEED_2_LITE:
            ranked, rerank_items, rerank_reasons = await self._reranker.rerank(
                request=request,
                image_url=image_url,
                ranked_hits=ranked,
                assets=assets,
            )
            reasons.extend(rerank_reasons)
        rerank_ms = _elapsed_ms(rerank_started)

        results = self._result_builder.build(
            ranked_hits=ranked,
            assets=assets,
            workspace_id=request.workspace_id,
            allowed_asset_types=tuple(item.value for item in request.filters.asset_type),
            top_k=request.top_k,
            rerank_items=rerank_items,
        )
        total_ms = _elapsed_ms(started)
        reasons = list(dict.fromkeys(reasons))

        capsule_id: str | None = None
        execution_id: str | None = None
        if self._history is not None:
            try:
                capsule_id, execution_id = await self._history.record_success(
                    request=request,
                    parsed_query=parsed_query,
                    results=results,
                    degraded=bool(reasons),
                    degraded_reasons=reasons,
                    latency_ms=round(total_ms),
                    existing_capsule_id=existing_capsule_id,
                )
            except Exception:
                if existing_capsule_id is not None:
                    raise
                logger.warning("search history persistence failed", exc_info=True)
                reasons.append("search completed but history persistence failed")

        timings = SearchTimings(
            parser_ms=parser_ms,
            embedding_ms=embedding_ms,
            recall_ms=recall_ms,
            fusion_ms=fusion_ms,
            rerank_ms=rerank_ms,
            hydration_ms=hydration_ms,
            total_ms=total_ms,
        )
        logger.info(
            "search completed workspace_id=%s query_type=%s results=%d "
            "channels=%d parser_ms=%.2f embedding_ms=%.2f recall_ms=%.2f "
            "fusion_ms=%.2f rerank_ms=%.2f hydration_ms=%.2f total_ms=%.2f degraded=%s",
            request.workspace_id,
            request.query_type.value,
            len(results),
            len(recall.channels),
            parser_ms,
            embedding_ms,
            recall_ms,
            fusion_ms,
            rerank_ms,
            hydration_ms,
            total_ms,
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
                query_image_upload_id=request.query_image_upload_id,
                precision_mode=request.precision_mode,
            ),
            parsed_query=parsed_query,
            fusion_method=request.fusion_method,
            rerank_method=request.rerank_method,
            search_engine_version=self._settings.search_engine_version,
            execution_id=execution_id,
            capsule_id=capsule_id,
            total=len(results),
            degraded=bool(reasons),
            degraded_reasons=reasons,
            timings=timings,
            results=results,
        )

    async def _resolve_image(self, request: SearchRequest) -> str | None:
        if request.query_image_url:
            return request.query_image_url
        if request.query_image_upload_id:
            if self._image_resolver is None:
                raise SearchUnavailableError("query image upload resolver is unavailable")
            try:
                return await self._image_resolver.resolve(
                    workspace_id=request.workspace_id,
                    upload_id=request.query_image_upload_id,
                )
            except Exception as exc:
                raise SearchUnavailableError("query image upload was not found") from exc
        return None


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
