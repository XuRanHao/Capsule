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
from capsule.vector_fusion import (
    DEFAULT_NATIVE_CONTENT_WEIGHT,
    fuse_native_dimension_vectors,
)

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
        operation_cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]] = {}
        outcomes = await asyncio.gather(
            *(
                self._embed_dimension(
                    request=request,
                    dimension=dimension,
                    image_url=resolved_image,
                    operation_cache=operation_cache,
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
        deduplicated_reasons = tuple(dict.fromkeys(reasons))
        return QueryEmbeddingPlan(
            vectors=tuple(vectors),
            degraded=bool(deduplicated_reasons),
            degraded_reasons=deduplicated_reasons,
        )

    async def _embed_dimension(
        self,
        *,
        request: SearchRequest,
        dimension: DimensionQuery,
        image_url: str | None,
        operation_cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]],
    ) -> tuple[QueryVector, str | None]:
        original_embedding = self._embed_original_query(
            request=request,
            image_url=image_url,
            operation_cache=operation_cache,
        )
        if dimension.embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
            original_result, fallback_reason = await original_embedding
            vector = self._normalize(original_result)
        else:
            (original_result, original_fallback), (
                dimension_result,
                dimension_fallback,
            ) = await asyncio.gather(
                original_embedding,
                self._embed_dimension_query(
                    request=request,
                    dimension=dimension,
                    image_url=image_url,
                    operation_cache=operation_cache,
                ),
            )
            vector = fuse_native_dimension_vectors(
                native_vector=self._normalize(original_result),
                dimension_vector=self._normalize(dimension_result),
                native_content_weight=DEFAULT_NATIVE_CONTENT_WEIGHT,
            )
            fallback_reason = original_fallback or dimension_fallback

        return (
            QueryVector(
                channel=dimension.embedding_type.value,
                embedding_type=dimension.embedding_type,
                vector=vector,
                weight=dimension.weight,
            ),
            fallback_reason,
        )

    async def _embed_original_query(
        self,
        *,
        request: SearchRequest,
        image_url: str | None,
        operation_cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]],
    ) -> tuple[EmbeddingResult, str | None]:
        operation_key: tuple[str, ...]
        if request.query_type is QueryType.IMAGE:
            if image_url is None:
                raise QueryEmbeddingError("image route requires a resolved image URL")
            operation_key = ("image", image_url)
        elif request.query_type is QueryType.IMAGE_TEXT:
            if image_url is None:
                raise QueryEmbeddingError("joint route requires a resolved image URL")
            assert request.query_text is not None
            operation_key = ("image_text", image_url, request.query_text)
        else:
            assert request.query_text is not None
            operation_key = ("text", request.query_text)
        return await self._embed_with_joint_fallback(
            operation_key=operation_key,
            image_url=image_url,
            operation_cache=operation_cache,
        )

    async def _embed_dimension_query(
        self,
        *,
        request: SearchRequest,
        dimension: DimensionQuery,
        image_url: str | None,
        operation_cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]],
    ) -> tuple[EmbeddingResult, str | None]:
        operation_key: tuple[str, ...]
        if dimension.source is QueryDimensionSource.IMAGE:
            if image_url is None:
                raise QueryEmbeddingError("image route requires a resolved image URL")
            operation_key = ("image", image_url)
        elif (
            dimension.source is QueryDimensionSource.JOINT
            and request.query_type is QueryType.IMAGE_TEXT
        ):
            if image_url is None:
                raise QueryEmbeddingError("joint route requires a resolved image URL")
            operation_key = ("image_text", image_url, dimension.query)
        elif request.query_type is QueryType.IMAGE and image_url is not None:
            operation_key = ("image", image_url)
        else:
            operation_key = ("text", dimension.query)

        return await self._embed_with_joint_fallback(
            operation_key=operation_key,
            image_url=image_url,
            operation_cache=operation_cache,
        )

    async def _embed_with_joint_fallback(
        self,
        *,
        operation_key: tuple[str, ...],
        image_url: str | None,
        operation_cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]],
    ) -> tuple[EmbeddingResult, str | None]:
        fallback_reason: str | None = None
        try:
            result = await self._cached_call(
                operation_cache,
                operation_key,
            )
        except Exception as exc:
            if operation_key[0] == "image_text" and image_url is not None:
                logger.warning(
                    "joint image_text embedding failed; image fallback used",
                    exc_info=True,
                )
                result = await self._cached_call(
                    operation_cache,
                    ("image", image_url),
                )
                fallback_reason = "joint image_text embedding failed; image fallback used"
            else:
                raise QueryEmbeddingError("query embedding operation failed") from exc
        return result, fallback_reason

    async def _call(self, operation: Awaitable[EmbeddingResult]) -> EmbeddingResult:
        async with self._semaphore:
            return await operation

    async def _cached_call(
        self,
        cache: dict[tuple[str, ...], asyncio.Task[EmbeddingResult]],
        key: tuple[str, ...],
    ) -> EmbeddingResult:
        task = cache.get(key)
        if task is None:
            operation_kind = key[0]
            if operation_kind == "text":
                operation = self._client.embed_text(key[1])
            elif operation_kind == "image":
                operation = self._client.embed_image(key[1])
            elif operation_kind == "image_text":
                operation = self._client.embed_image_text(key[1], key[2])
            else:
                raise QueryEmbeddingError(
                    f"unsupported query embedding operation: {operation_kind}"
                )
            task = asyncio.create_task(self._call(operation))
            cache[key] = task
        return await task

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
