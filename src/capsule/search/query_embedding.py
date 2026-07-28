import asyncio
import logging
import math
from collections.abc import Awaitable

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.schemas import EmbeddingResult
from capsule.search.contracts import QueryEmbeddingClient
from capsule.search.models import (
    DimensionQuery,
    ParsedQuery,
    QueryDimensionSource,
    QueryEmbeddingPlan,
    QueryType,
    QueryVector,
    SearchRequest,
)
from capsule.search.query_parser import QueryParser

logger = logging.getLogger(__name__)


class QueryEmbeddingError(RuntimeError):
    pass


class QueryEmbeddingService:
    """Generate every enabled query route concurrently in one vector space."""

    def __init__(
        self,
        client: QueryEmbeddingClient,
        settings: Settings,
        *,
        expected_dimension: int | None = None,
    ) -> None:
        self._client = client
        self._expected_dimension = expected_dimension or settings.embedding_dimension
        self._semaphore = asyncio.Semaphore(settings.search_embedding_concurrency)

    async def embed(
        self,
        request: SearchRequest,
        parsed_query: ParsedQuery | None = None,
        *,
        image_url: str | None = None,
    ) -> QueryEmbeddingPlan:
        parsed = parsed_query or (await QueryParser().parse(request, image_url=image_url))[0]
        resolved_image = image_url or request.query_image_url
        outcomes = await asyncio.gather(
            *(
                self._embed_dimension(
                    request=request,
                    dimension=dimension,
                    image_url=resolved_image,
                )
                for dimension in parsed.dimension_queries
            ),
            return_exceptions=True,
        )

        vectors: list[QueryVector] = []
        reasons: list[str] = []
        for dimension, outcome in zip(parsed.dimension_queries, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                reason = f"query embedding route {dimension.embedding_type.value} failed"
                reasons.append(reason)
                logger.warning(reason, exc_info=outcome)
                continue
            vector, fallback_reason = outcome
            vectors.append(vector)
            if fallback_reason:
                reasons.append(fallback_reason)
        if not vectors:
            raise QueryEmbeddingError("all query embedding routes failed")

        remaining_weight = sum(item.weight for item in vectors)
        if remaining_weight <= 0:
            raise QueryEmbeddingError("query embedding route weight must be positive")
        if abs(remaining_weight - 1.0) > 1e-6:
            vectors = [
                QueryVector(
                    channel=item.channel,
                    embedding_type=item.embedding_type,
                    vector=item.vector,
                    weight=item.weight / remaining_weight,
                )
                for item in vectors
            ]
            reasons.append("failed query routes were removed and weights were renormalized")
        return QueryEmbeddingPlan(
            vectors=tuple(vectors),
            degraded=bool(reasons),
            degraded_reasons=tuple(reasons),
        )

    async def _embed_dimension(
        self,
        *,
        request: SearchRequest,
        dimension: DimensionQuery,
        image_url: str | None,
    ) -> tuple[QueryVector, str | None]:
        operation: Awaitable[EmbeddingResult]
        if dimension.embedding_type is not EmbeddingType.NATIVE_MULTIMODAL:
            operation = self._client.embed_text(dimension.query)
        elif dimension.source is QueryDimensionSource.IMAGE:
            if image_url is None:
                raise QueryEmbeddingError("image route requires a resolved image URL")
            operation = self._client.embed_image(image_url)
        elif (
            dimension.source is QueryDimensionSource.JOINT
            and request.query_type is QueryType.IMAGE_TEXT
        ):
            if image_url is None:
                raise QueryEmbeddingError("joint route requires a resolved image URL")
            operation = self._client.embed_image_text(image_url, dimension.query)
        elif request.query_type is QueryType.IMAGE and image_url is not None:
            operation = self._client.embed_image(image_url)
        else:
            operation = self._client.embed_text(dimension.query)

        fallback_reason: str | None = None
        try:
            result = await self._call(operation)
        except Exception as exc:
            if (
                dimension.embedding_type is EmbeddingType.NATIVE_MULTIMODAL
                and dimension.source is QueryDimensionSource.JOINT
                and image_url is not None
            ):
                logger.warning(
                    "joint image_text embedding failed; image fallback used",
                    exc_info=True,
                )
                result = await self._call(self._client.embed_image(image_url))
                fallback_reason = "joint image_text embedding failed; image fallback used"
            else:
                raise QueryEmbeddingError(
                    f"{dimension.embedding_type.value} query embedding failed"
                ) from exc

        return (
            QueryVector(
                channel=dimension.embedding_type.value,
                embedding_type=dimension.embedding_type,
                vector=self._normalize(result),
                weight=dimension.weight,
            ),
            fallback_reason,
        )

    async def _call(self, operation: Awaitable[EmbeddingResult]) -> EmbeddingResult:
        async with self._semaphore:
            return await operation

    def _normalize(self, result: EmbeddingResult) -> list[float]:
        if len(result.vector) != self._expected_dimension:
            raise QueryEmbeddingError(
                "query embedding dimension mismatch: "
                f"expected {self._expected_dimension}, got {len(result.vector)}"
            )
        if any(not math.isfinite(value) for value in result.vector):
            raise QueryEmbeddingError("query embedding contains NaN or infinity")
        norm = math.sqrt(sum(value * value for value in result.vector))
        if norm == 0:
            raise QueryEmbeddingError("query embedding must not be all zeros")
        return [value / norm for value in result.vector]
