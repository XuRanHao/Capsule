import logging
import math
from collections.abc import Mapping

from capsule.enums import EmbeddingType
from capsule.search.contracts import SearchUnderstandingClient
from capsule.search.models import (
    DimensionQuery,
    ParsedQuery,
    QueryDimensionSource,
    QueryEnhancement,
    QueryType,
    SearchRequest,
)

logger = logging.getLogger(__name__)

_ENHANCEMENT_FALLBACK_REASON = "query enhancement fallback used"


class QueryParser:
    """Build query routes and enhance every selected text-bearing dimension."""

    def __init__(self, client: SearchUnderstandingClient | None = None) -> None:
        self._client = client

    async def parse(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> tuple[ParsedQuery, tuple[str, ...]]:
        del image_url
        fallback = _fallback_query_plan(request)
        if not _should_enhance(request):
            return fallback, ()

        try:
            if self._client is None:
                raise RuntimeError("query enhancer is unavailable")
            enhancement = await self._client.enhance_search_query(
                query_text=request.query_text or "",
                embedding_types=request.embedding_types,
            )
            queries = _validate_queries(
                enhancement,
                requested_types=request.embedding_types,
            )
            weights = _validate_and_normalize_weights(
                enhancement,
                requested_types=request.embedding_types,
            )
            return _build_query_plan(request, queries=queries, weights=weights), ()
        except Exception:
            logger.warning(
                "query enhancement failed; original-query fallback used",
                exc_info=True,
            )
            return fallback, (_ENHANCEMENT_FALLBACK_REASON,)


def _should_enhance(request: SearchRequest) -> bool:
    return (
        request.query_type in {QueryType.TEXT, QueryType.IMAGE_TEXT}
        and len(request.embedding_types) > 1
    )


def _validate_queries(
    enhancement: QueryEnhancement,
    *,
    requested_types: list[EmbeddingType],
) -> dict[EmbeddingType, str]:
    if set(enhancement.queries) != set(requested_types):
        raise ValueError("enhanced queries must exactly match requested embedding types")
    queries: dict[EmbeddingType, str] = {}
    for embedding_type in requested_types:
        query = enhancement.queries[embedding_type].strip()
        if not query:
            raise ValueError("enhanced queries must not be empty")
        queries[embedding_type] = query
    return queries


def _validate_and_normalize_weights(
    enhancement: QueryEnhancement,
    *,
    requested_types: list[EmbeddingType],
) -> dict[EmbeddingType, float]:
    weights = enhancement.weights
    if set(weights) != set(requested_types):
        raise ValueError("enhanced weights must exactly match requested embedding types")
    validated: dict[EmbeddingType, float] = {}
    for embedding_type in requested_types:
        value = weights[embedding_type]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("enhanced weights must be numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError("enhanced weights must be positive finite numbers")
        validated[embedding_type] = numeric_value
    total = sum(validated.values())
    return {
        embedding_type: value / total
        for embedding_type, value in validated.items()
    }


def _fallback_query_plan(request: SearchRequest) -> ParsedQuery:
    query = request.query_text or "参考图片"
    queries = {embedding_type: query for embedding_type in request.embedding_types}
    weight = 1.0 / len(request.embedding_types)
    weights = {embedding_type: weight for embedding_type in request.embedding_types}
    return _build_query_plan(request, queries=queries, weights=weights)


def _build_query_plan(
    request: SearchRequest,
    *,
    queries: Mapping[EmbeddingType, str],
    weights: Mapping[EmbeddingType, float],
) -> ParsedQuery:
    return ParsedQuery(
        dimension_queries=[
            DimensionQuery(
                embedding_type=embedding_type,
                query=queries[embedding_type],
                weight=weights[embedding_type],
                source=_source_for(request.query_type, embedding_type),
            )
            for embedding_type in request.embedding_types
        ]
    )


def _source_for(
    query_type: QueryType,
    embedding_type: EmbeddingType,
) -> QueryDimensionSource:
    if query_type is QueryType.IMAGE:
        return QueryDimensionSource.IMAGE
    if (
        query_type is QueryType.IMAGE_TEXT
        and embedding_type is EmbeddingType.NATIVE_MULTIMODAL
    ):
        return QueryDimensionSource.JOINT
    return QueryDimensionSource.TEXT
