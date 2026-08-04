import logging

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


class QueryParser:
    """Turn a user query into explicit, weighted embedding routes.

    Model parsing is intentionally isolated from deterministic fallbacks so a
    transient understanding-model failure never makes the whole search API
    unavailable.
    """

    def __init__(self, client: SearchUnderstandingClient | None = None) -> None:
        self._client = client

    async def parse(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> tuple[ParsedQuery, tuple[str, ...]]:
        native_only = request.embedding_types == [EmbeddingType.NATIVE_MULTIMODAL]
        if not request.precision_mode or native_only:
            if request.query_type is QueryType.IMAGE:
                return _select_requested_dimensions(_quick_image_query(), request), ()
            return _select_requested_dimensions(_quick_semantic_query(request), request), ()
        if self._client is not None:
            try:
                parsed = await self._client.parse_search_query(
                    request,
                    image_url=image_url,
                )
                return _select_requested_dimensions(
                    _normalize_weights(parsed),
                    request,
                    preserve_weights=request.query_type
                    in {QueryType.TEXT, QueryType.IMAGE_TEXT},
                ), ()
            except Exception:
                logger.warning("query parser model call failed; fallback used", exc_info=True)
        return _select_requested_dimensions(
            _fallback_query(request),
            request,
        ), ("query parser fallback used",)


def _quick_semantic_query(request: SearchRequest) -> ParsedQuery:
    """Route ordinary text and image-text searches without a model round trip."""
    return _fallback_query(request)


def _quick_image_query() -> ParsedQuery:
    return ParsedQuery(
        dimension_queries=[
            DimensionQuery(
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                query="参考图片",
                weight=1.0,
                source=QueryDimensionSource.IMAGE,
            )
        ],
    )


def _fallback_query(request: SearchRequest) -> ParsedQuery:
    if request.query_type is QueryType.IMAGE:
        return ParsedQuery(
            dimension_queries=[
                DimensionQuery(
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                    query="参考图片",
                    weight=1.0,
                    source=QueryDimensionSource.IMAGE,
                )
            ],
        )

    text = request.query_text or "参考图片"
    if request.query_type is QueryType.IMAGE_TEXT:
        return ParsedQuery(
            dimension_queries=[
                DimensionQuery(
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                    query=text,
                    weight=0.35,
                    source=QueryDimensionSource.JOINT,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.ASSET_DESCRIPTION,
                    query=text,
                    weight=0.20,
                    source=QueryDimensionSource.TEXT,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.SUBJECT_CONTENT,
                    query=text,
                    weight=0.15,
                    source=QueryDimensionSource.TEXT,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.VISUAL_STYLE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.TARGET_AUDIENCE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                ),
            ],
        )

    return ParsedQuery(
        dimension_queries=[
            DimensionQuery(
                embedding_type=EmbeddingType.ASSET_DESCRIPTION,
                query=text,
                weight=0.35,
            ),
            DimensionQuery(
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                query=text,
                weight=0.35,
            ),
            DimensionQuery(
                embedding_type=EmbeddingType.SUBJECT_CONTENT,
                query=text,
                weight=0.15,
            ),
            DimensionQuery(
                embedding_type=EmbeddingType.SCENE_THEME,
                query=text,
                weight=0.05,
            ),
            DimensionQuery(
                embedding_type=EmbeddingType.VISUAL_STYLE,
                query=text,
                weight=0.05,
            ),
            DimensionQuery(
                embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
                query=text,
                weight=0.05,
            ),
        ],
    )


def _normalize_weights(parsed: ParsedQuery) -> ParsedQuery:
    total = sum(item.weight for item in parsed.dimension_queries)
    if abs(total - 1.0) <= 1e-9:
        return parsed
    return parsed.model_copy(
        update={
            "dimension_queries": [
                item.model_copy(update={"weight": item.weight / total})
                for item in parsed.dimension_queries
            ]
        }
    )


def _select_requested_dimensions(
    parsed: ParsedQuery,
    request: SearchRequest,
    *,
    preserve_weights: bool = False,
) -> ParsedQuery:
    """Keep user-selected channels and optionally preserve model intent weights."""

    parsed_by_type = {
        dimension.embedding_type: dimension for dimension in parsed.dimension_queries
    }
    all_dimensions_were_parsed = all(
        embedding_type in parsed_by_type for embedding_type in request.embedding_types
    )
    dimensions = []
    for embedding_type in request.embedding_types:
        dimension = parsed_by_type.get(embedding_type) or _default_dimension(
            request,
            embedding_type,
        )
        dimensions.append(dimension)
    if preserve_weights and all_dimensions_were_parsed:
        total_weight = sum(dimension.weight for dimension in dimensions)
        dimensions = [
            dimension.model_copy(update={"weight": dimension.weight / total_weight})
            for dimension in dimensions
        ]
    else:
        equal_weight = 1.0 / len(dimensions)
        dimensions = [
            dimension.model_copy(update={"weight": equal_weight})
            for dimension in dimensions
        ]
    return parsed.model_copy(update={"dimension_queries": dimensions})


def _default_dimension(
    request: SearchRequest,
    embedding_type: EmbeddingType,
) -> DimensionQuery:
    if request.query_type is QueryType.IMAGE:
        source = QueryDimensionSource.IMAGE
    elif (
        request.query_type is QueryType.IMAGE_TEXT
        and embedding_type is EmbeddingType.NATIVE_MULTIMODAL
    ):
        source = QueryDimensionSource.JOINT
    else:
        source = QueryDimensionSource.TEXT
    return DimensionQuery(
        embedding_type=embedding_type,
        query=request.query_text or "参考图片",
        weight=1.0,
        source=source,
    )
