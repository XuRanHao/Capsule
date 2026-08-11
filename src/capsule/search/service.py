import logging
from dataclasses import replace
from time import perf_counter

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.search.contracts import (
    AssetSearchRepository,
    ClusterSearchRepository,
    QueryImageResolver,
    SearchVectorIndexPreparer,
    TextSearchRepository,
)
from capsule.search.fusion import FusionEngine
from capsule.search.history import SearchHistoryRepository
from capsule.search.models import (
    QueryEmbeddingPlan,
    SearchDimensionSuggestionRequest,
    SearchDimensionSuggestionResponse,
    SearchQueryEcho,
    SearchRequest,
    SearchResponse,
    SearchTimings,
)
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
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
        query_parser: QueryParser | None = None,
        clusters: ClusterSearchRepository | None = None,
        history: SearchHistoryRepository | None = None,
        image_resolver: QueryImageResolver | None = None,
        search_vector_preparer: SearchVectorIndexPreparer | None = None,
        text_recall: TextSearchRepository | None = None,
    ) -> None:
        self._query_embedding = query_embedding
        self._query_parser = query_parser or QueryParser()
        self._recall = recall
        self._assets = assets
        self._fusion = FusionEngine(
            rrf_k=settings.search_rrf_k,
            candidate_cap=settings.search_candidate_cap,
        )
        self._result_builder = SearchResultBuilder(
            same_source_limit=settings.search_same_source_limit
        )
        self._clusters = clusters
        self._history = history
        self._image_resolver = image_resolver
        self._search_vector_preparer = search_vector_preparer
        self._text_recall = text_recall
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

        query_enhancement_started = perf_counter()
        parsed_query, query_enhancement_reasons = await self._query_parser.parse(
            request,
            image_url=image_url,
        )
        reasons.extend(query_enhancement_reasons)
        query_enhancement_ms = _elapsed_ms(query_enhancement_started)

        embedding_started = perf_counter()
        plan = await self._query_embedding.embed(
            request,
            parsed_query,
            image_url=image_url,
        )
        plan = await self._prepare_search_vector_indexes(
            plan=plan,
            workspace_id=request.workspace_id,
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
        text_hits = []
        if self._text_recall is not None and request.query_text:
            try:
                text_hits = list(
                    await self._text_recall.search_text(
                        workspace_id=request.workspace_id,
                        query_text=request.query_text,
                        filters=request.filters,
                        created_by=request.created_by,
                        limit=min(
                            request.top_k * self._settings.search_channel_top_k_multiplier,
                            self._settings.search_channel_top_k_cap,
                        ),
                    )
                )
            except Exception:
                logger.warning("local text recall failed", exc_info=True)
                reasons.append("local text recall failed")
        recall_ms = _elapsed_ms(recall_started)
        if not recall.channels and not text_hits:
            raise SearchUnavailableError("all search recall routes failed")
        if not recall.channels:
            reasons.append("all Milvus recall channels failed")

        fusion_started = perf_counter()
        ranked = self._fusion.fuse(
            recall.channels,
            request.fusion_method,
            text_hits=text_hits,
        )
        fusion_ms = _elapsed_ms(fusion_started)

        hydration_started = perf_counter()
        assets = await self._assets.get_by_ids(
            workspace_id=request.workspace_id,
            asset_ids=[item.asset_id for item in ranked],
            embedding_ids=[
                match.embedding_id
                for item in ranked
                for match in item.matched_channels
                if match.embedding_id is not None
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

        results = self._result_builder.build(
            ranked_hits=ranked,
            assets=assets,
            workspace_id=request.workspace_id,
            allowed_asset_types=tuple(item.value for item in request.filters.asset_type),
            top_k=request.top_k,
        )
        parent_asset_ids = list(
            dict.fromkeys(
                result.parent_asset_id
                for result in results
                if result.index_role == "child" and result.parent_asset_id is not None
            )
        )
        if parent_asset_ids:
            try:
                siblings = await self._assets.get_children_by_parent_ids(
                    workspace_id=request.workspace_id,
                    parent_asset_ids=parent_asset_ids,
                )
                results = self._result_builder.expand_adjacent_children(
                    results,
                    recalled_assets=assets,
                    sibling_assets=siblings,
                )
            except Exception:
                logger.warning("adjacent document context expansion failed", exc_info=True)
                reasons.append("adjacent document context expansion failed")
        hydration_ms = _elapsed_ms(hydration_started)
        cluster_started = perf_counter()
        cluster_results = []
        if self._clusters is not None and results:
            ranked_scores = {item.asset_id: item.score for item in ranked}
            asset_scores = {
                asset_id: ranked_scores[asset_id]
                for result in results
                for asset_id in result.folded_asset_ids
                if asset_id in ranked_scores
            }
            try:
                cluster_results = list(
                    await self._clusters.search_by_assets(
                        workspace_id=request.workspace_id,
                        asset_scores=asset_scores,
                        embedding_types=tuple(
                            dict.fromkeys(vector.embedding_type.value for vector in plan.vectors)
                        ),
                        limit=min(request.top_k, self._settings.search_cluster_top_k),
                    )
                )
            except Exception:
                logger.warning("cluster result aggregation failed", exc_info=True)
                reasons.append("cluster result aggregation failed")
        cluster_ms = _elapsed_ms(cluster_started)
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
            query_enhancement_ms=query_enhancement_ms,
            embedding_ms=embedding_ms,
            recall_ms=recall_ms,
            fusion_ms=fusion_ms,
            hydration_ms=hydration_ms,
            cluster_ms=cluster_ms,
            total_ms=total_ms,
        )
        logger.info(
            "search completed workspace_id=%s query_type=%s results=%d "
            "vector_channels=%d text_hits=%d query_enhancement_ms=%.2f "
            "embedding_ms=%.2f recall_ms=%.2f "
            "fusion_ms=%.2f hydration_ms=%.2f "
            "cluster_ms=%.2f total_ms=%.2f degraded=%s",
            request.workspace_id,
            request.query_type.value,
            len(results),
            len(recall.channels),
            len(text_hits),
            query_enhancement_ms,
            embedding_ms,
            recall_ms,
            fusion_ms,
            hydration_ms,
            cluster_ms,
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
                embedding_types=request.embedding_types,
            ),
            parsed_query=parsed_query,
            fusion_method=request.fusion_method,
            search_engine_version=self._settings.search_engine_version,
            execution_id=execution_id,
            capsule_id=capsule_id,
            total=len(results),
            asset_total=len(results),
            cluster_total=len(cluster_results),
            degraded=bool(reasons),
            degraded_reasons=reasons,
            timings=timings,
            assets=results,
            clusters=cluster_results,
            results=results,
        )

    async def suggest_dimensions(
        self,
        request: SearchDimensionSuggestionRequest,
    ) -> SearchDimensionSuggestionResponse:
        suggestion = await self._query_parser.suggest_dimensions(
            query_text=request.query_text,
            asset_types=request.asset_types,
        )
        return suggestion

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

    async def _prepare_search_vector_indexes(
        self,
        *,
        plan: QueryEmbeddingPlan,
        workspace_id: str,
    ) -> QueryEmbeddingPlan:
        preparer = self._search_vector_preparer
        if preparer is None:
            return plan

        dimension_types = tuple(
            dict.fromkeys(
                vector.embedding_type
                for vector in plan.vectors
                if vector.embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
            )
        )
        if not dimension_types:
            return plan

        try:
            errors = await preparer.ensure_search_vectors(
                workspace_id=workspace_id,
                embedding_types=dimension_types,
            )
        except Exception:
            logger.warning(
                "search vector index preparation failed for all requested dimensions",
                exc_info=True,
            )
            errors = {
                embedding_type: "search vector index preparer failed"
                for embedding_type in dimension_types
            }

        failed_types = set(dimension_types).intersection(errors)
        preparation_reasons = [
            f"search vector index preparation for {embedding_type.value} failed"
            for embedding_type in dimension_types
            if embedding_type in failed_types
        ]
        for embedding_type in dimension_types:
            if embedding_type in failed_types:
                logger.warning(
                    "search vector index preparation for %s failed: %s",
                    embedding_type.value,
                    errors[embedding_type],
                )

        if not failed_types:
            return plan

        remaining = tuple(
            vector for vector in plan.vectors if vector.embedding_type not in failed_types
        )
        if not remaining:
            raise SearchUnavailableError("all search vector index preparations failed")

        weight_total = sum(vector.weight for vector in remaining)
        if weight_total <= 0:
            raise SearchUnavailableError("remaining search channel weight must be positive")
        normalized = tuple(
            replace(vector, weight=vector.weight / weight_total) for vector in remaining
        )
        degraded_reasons = tuple(dict.fromkeys((*plan.degraded_reasons, *preparation_reasons)))
        return replace(
            plan,
            vectors=normalized,
            degraded=True,
            degraded_reasons=degraded_reasons,
        )


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
