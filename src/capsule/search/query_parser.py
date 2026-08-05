import logging
import math
from collections.abc import Mapping

from capsule.enums import EmbeddingType
from capsule.search.contracts import SearchUnderstandingClient
from capsule.search.models import (
    DimensionQuery,
    ParsedQuery,
    QueryDimensionSource,
    QueryType,
    SearchRequest,
)

logger = logging.getLogger(__name__)

_WEIGHT_INTENT_TERMS = (
    "重点",
    "优先",
    "更看重",
    "侧重",
    "其次",
    "为主",
    "权重",
)
_PRIMARY_INTENT_PHRASES = (
    "主要看",
    "主要关注",
    "主要匹配",
    "主要考虑",
    "主要检索",
)
_WEIGHT_FALLBACK_REASON = "query weight resolution fallback used"


class QueryParser:
    """Build a deterministic query plan and resolve explicit text preferences."""

    def __init__(self, client: SearchUnderstandingClient | None = None) -> None:
        self._client = client

    async def parse(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> tuple[ParsedQuery, tuple[str, ...]]:
        del image_url
        equal_weights = _equal_weights(request.embedding_types)
        if not _should_resolve_weights(request):
            return _build_query_plan(request, equal_weights), ()

        try:
            if self._client is None:
                raise RuntimeError("query weight resolver is unavailable")
            weights = await self._client.resolve_query_weights(
                query_text=request.query_text or "",
                embedding_types=request.embedding_types,
            )
            normalized = _validate_and_normalize_weights(
                weights,
                requested_types=request.embedding_types,
            )
            return _build_query_plan(request, normalized), ()
        except Exception:
            logger.warning(
                "query weight resolution failed; equal-weight fallback used",
                exc_info=True,
            )
            return _build_query_plan(request, equal_weights), (_WEIGHT_FALLBACK_REASON,)


def _should_resolve_weights(request: SearchRequest) -> bool:
    query_text = request.query_text
    return (
        request.query_type is not QueryType.IMAGE
        and len(request.embedding_types) > 1
        and query_text is not None
        and (
            any(term in query_text for term in _WEIGHT_INTENT_TERMS)
            or any(phrase in query_text for phrase in _PRIMARY_INTENT_PHRASES)
        )
    )


def _equal_weights(
    embedding_types: list[EmbeddingType],
) -> dict[EmbeddingType, float]:
    weight = 1.0 / len(embedding_types)
    return {embedding_type: weight for embedding_type in embedding_types}


def _validate_and_normalize_weights(
    weights: Mapping[EmbeddingType, float],
    *,
    requested_types: list[EmbeddingType],
) -> dict[EmbeddingType, float]:
    if set(weights) != set(requested_types):
        raise ValueError("resolved weights must exactly match requested embedding types")
    validated: dict[EmbeddingType, float] = {}
    for embedding_type in requested_types:
        value = weights[embedding_type]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("resolved weights must be numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError("resolved weights must be positive finite numbers")
        validated[embedding_type] = numeric_value
    total = sum(validated.values())
    return {
        embedding_type: value / total
        for embedding_type, value in validated.items()
    }


def _build_query_plan(
    request: SearchRequest,
    weights: Mapping[EmbeddingType, float],
) -> ParsedQuery:
    query = request.query_text or "参考图片"
    return ParsedQuery(
        dimension_queries=[
            DimensionQuery(
                embedding_type=embedding_type,
                query=query,
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
