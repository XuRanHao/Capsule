import logging

from capsule.enums import EmbeddingType
from capsule.search.contracts import SearchUnderstandingClient
from capsule.search.models import (
    DimensionQuery,
    ParsedQuery,
    QueryConstraint,
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
        if request.query_type is QueryType.IMAGE and not request.precision_mode:
            return _quick_image_query(), ()
        if self._client is not None:
            try:
                parsed = await self._client.parse_search_query(
                    request,
                    image_url=image_url,
                )
                return _normalize_weights(parsed), ()
            except Exception:
                logger.warning("query parser model call failed; fallback used", exc_info=True)
        return _fallback_query(request), ("query parser fallback used",)


def _quick_image_query() -> ParsedQuery:
    return ParsedQuery(
        query_summary="按参考图片进行原生多模态快速检索",
        dimension_queries=[
            DimensionQuery(
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                query="参考图片",
                weight=1.0,
                source=QueryDimensionSource.IMAGE,
                constraint=QueryConstraint.MATCH,
            )
        ],
        parser_mode="deterministic_quick",
    )


def _fallback_query(request: SearchRequest) -> ParsedQuery:
    if request.query_type is QueryType.IMAGE:
        return ParsedQuery(
            query_summary="图片精搜降级为原生多模态检索",
            dimension_queries=[
                DimensionQuery(
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                    query="参考图片",
                    weight=1.0,
                    source=QueryDimensionSource.IMAGE,
                )
            ],
            parser_mode="fallback",
        )

    text = request.query_text or "参考图片"
    negative_terms = _extract_negative_terms(text)
    if request.query_type is QueryType.IMAGE_TEXT:
        return ParsedQuery(
            query_summary=text,
            dimension_queries=[
                DimensionQuery(
                    embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                    query=text,
                    weight=0.35,
                    source=QueryDimensionSource.JOINT,
                    constraint=QueryConstraint.MAINTAIN,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.ASSET_DESCRIPTION,
                    query=text,
                    weight=0.20,
                    source=QueryDimensionSource.TEXT,
                    constraint=QueryConstraint.MODIFY,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.SUBJECT_CONTENT,
                    query=text,
                    weight=0.15,
                    source=QueryDimensionSource.TEXT,
                    constraint=QueryConstraint.MAINTAIN,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.VISUAL_STYLE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                    constraint=QueryConstraint.MODIFY,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                    constraint=QueryConstraint.MODIFY,
                ),
                DimensionQuery(
                    embedding_type=EmbeddingType.TARGET_AUDIENCE,
                    query=text,
                    weight=0.10,
                    source=QueryDimensionSource.TEXT,
                    constraint=QueryConstraint.MODIFY,
                ),
            ],
            negative_terms=negative_terms,
            parser_mode="fallback",
        )

    return ParsedQuery(
        query_summary=text,
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
        negative_terms=negative_terms,
        parser_mode="fallback",
    )


def _extract_negative_terms(text: str) -> list[str]:
    markers = ("排除", "不要", "不含", "去掉")
    terms: list[str] = []
    for marker in markers:
        if marker not in text:
            continue
        suffix = text.split(marker, 1)[1].strip(" ：:，,。")
        if suffix:
            terms.append(suffix[:100])
    return terms


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
