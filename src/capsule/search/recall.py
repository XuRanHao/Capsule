import asyncio
import logging
from time import perf_counter

from capsule.config import Settings
from capsule.search.contracts import VectorSearchRepository
from capsule.search.models import (
    ChannelRecall,
    QueryEmbeddingPlan,
    QueryVector,
    RecallBatch,
    SearchFilters,
)

logger = logging.getLogger(__name__)


class MultiChannelRecall:
    def __init__(self, repository: VectorSearchRepository, settings: Settings) -> None:
        self._repository = repository
        self._settings = settings

    async def search(
        self,
        *,
        plan: QueryEmbeddingPlan,
        workspace_id: str,
        filters: SearchFilters,
        top_k: int,
    ) -> RecallBatch:
        channel_limit = min(
            top_k * self._settings.search_channel_top_k_multiplier,
            self._settings.search_channel_top_k_cap,
        )
        outcomes = await asyncio.gather(
            *(
                self._search_channel(
                    query_vector=query_vector,
                    workspace_id=workspace_id,
                    filters=filters,
                    limit=channel_limit,
                )
                for query_vector in plan.vectors
            ),
            return_exceptions=True,
        )
        recalls: list[ChannelRecall] = []
        reasons: list[str] = []
        for query_vector, outcome in zip(plan.vectors, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                reason = f"recall channel {query_vector.channel} failed"
                reasons.append(reason)
                logger.warning(reason, exc_info=outcome)
                continue
            recalls.append(outcome)
        return RecallBatch(
            channels=tuple(recalls),
            degraded=bool(reasons),
            degraded_reasons=tuple(reasons),
        )

    async def _search_channel(
        self,
        *,
        query_vector: QueryVector,
        workspace_id: str,
        filters: SearchFilters,
        limit: int,
    ) -> ChannelRecall:
        started = perf_counter()
        hits = await self._repository.search(
            vector=query_vector.vector,
            workspace_id=workspace_id,
            embedding_type=query_vector.embedding_type.value,
            filters=filters,
            limit=limit,
        )
        logger.info(
            "recall channel completed workspace_id=%s channel=%s hits=%d latency_ms=%.2f",
            workspace_id,
            query_vector.channel,
            len(hits),
            (perf_counter() - started) * 1000,
        )
        return ChannelRecall(query_vector=query_vector, hits=tuple(hits))
