import asyncio
import logging
import math
from collections.abc import Awaitable

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.schemas import EmbeddingResult
from capsule.search.contracts import QueryEmbeddingClient
from capsule.search.models import (
    QueryEmbeddingPlan,
    QueryType,
    QueryVector,
    SearchRequest,
)

logger = logging.getLogger(__name__)


class QueryEmbeddingError(RuntimeError):
    pass


class QueryEmbeddingService:
    """Build query vectors while sharing a bounded client across requests."""

    def __init__(
        self,
        client: QueryEmbeddingClient,
        settings: Settings,
        *,
        expected_dimension: int | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._expected_dimension = expected_dimension or settings.embedding_dimension
        self._semaphore = asyncio.Semaphore(settings.search_embedding_concurrency)

    async def embed(self, request: SearchRequest) -> QueryEmbeddingPlan:
        if request.query_type is QueryType.TEXT:
            return await self._embed_text_query(request)
        if request.query_type is QueryType.IMAGE:
            return await self._embed_image_query(request)
        return await self._embed_image_text_query(request)

    async def _embed_text_query(self, request: SearchRequest) -> QueryEmbeddingPlan:
        assert request.query_text is not None
        try:
            result = await self._call(self._client.embed_text(request.query_text))
        except Exception as exc:
            raise QueryEmbeddingError("text query embedding failed") from exc
        vector = self._normalize(result)
        return QueryEmbeddingPlan(
            vectors=(
                self._query_vector(
                    "native_multimodal",
                    EmbeddingType.NATIVE_MULTIMODAL,
                    vector,
                    self._settings.search_native_weight,
                ),
                self._query_vector(
                    "asset_description",
                    EmbeddingType.ASSET_DESCRIPTION,
                    vector,
                    self._settings.search_description_weight,
                ),
                self._query_vector(
                    "subject_content",
                    EmbeddingType.SUBJECT_CONTENT,
                    vector,
                    self._settings.search_subject_weight,
                ),
                self._query_vector(
                    "visual_style",
                    EmbeddingType.VISUAL_STYLE,
                    vector,
                    self._settings.search_visual_weight,
                ),
            )
        )

    async def _embed_image_query(self, request: SearchRequest) -> QueryEmbeddingPlan:
        assert request.query_image_url is not None
        try:
            result = await self._call(self._client.embed_image(request.query_image_url))
        except Exception as exc:
            raise QueryEmbeddingError("image query embedding failed") from exc
        return QueryEmbeddingPlan(
            vectors=(
                self._query_vector(
                    "native_multimodal",
                    EmbeddingType.NATIVE_MULTIMODAL,
                    self._normalize(result),
                    self._settings.search_native_weight,
                ),
            )
        )

    async def _embed_image_text_query(self, request: SearchRequest) -> QueryEmbeddingPlan:
        assert request.query_image_url is not None
        assert request.query_text is not None

        combined, text = await asyncio.gather(
            self._call(
                self._client.embed_image_text(
                    request.query_image_url,
                    request.query_text,
                )
            ),
            self._call(self._client.embed_text(request.query_text)),
            return_exceptions=True,
        )
        reasons: list[str] = []
        vectors: list[QueryVector] = []

        text_vector: list[float] | None = None
        if isinstance(text, BaseException):
            reasons.append("image_text semantic text embedding failed")
            logger.warning("image_text semantic text embedding failed", exc_info=text)
        else:
            try:
                text_vector = self._normalize(text)
            except QueryEmbeddingError:
                reasons.append("image_text semantic text embedding is invalid")
                logger.warning("image_text semantic text embedding is invalid", exc_info=True)

        combined_vector: list[float] | None = None
        if isinstance(combined, BaseException):
            logger.warning("joint image_text embedding failed", exc_info=combined)
        else:
            try:
                combined_vector = self._normalize(combined)
            except QueryEmbeddingError:
                logger.warning("joint image_text embedding is invalid", exc_info=True)

        if combined_vector is None:
            reasons.append("joint image_text embedding unavailable; separate-vector fallback used")
            image = await self._embed_image_fallback(request.query_image_url)
            if image is not None and text_vector is not None:
                vectors.extend(
                    [
                        self._query_vector(
                            "native_multimodal:image",
                            EmbeddingType.NATIVE_MULTIMODAL,
                            image,
                            self._settings.search_native_weight / 2,
                        ),
                        self._query_vector(
                            "native_multimodal:text",
                            EmbeddingType.NATIVE_MULTIMODAL,
                            text_vector,
                            self._settings.search_native_weight / 2,
                        ),
                    ]
                )
            elif image is not None:
                vectors.append(
                    self._query_vector(
                        "native_multimodal:image",
                        EmbeddingType.NATIVE_MULTIMODAL,
                        image,
                        self._settings.search_native_weight,
                    )
                )
            elif text_vector is not None:
                reasons.append("image fallback embedding failed; text-only fallback used")
                vectors.append(
                    self._query_vector(
                        "native_multimodal:text",
                        EmbeddingType.NATIVE_MULTIMODAL,
                        text_vector,
                        self._settings.search_native_weight,
                    )
                )
        else:
            vectors.append(
                self._query_vector(
                    "native_multimodal",
                    EmbeddingType.NATIVE_MULTIMODAL,
                    combined_vector,
                    self._settings.search_native_weight,
                )
            )

        if text_vector is not None:
            vectors.extend(self._semantic_text_vectors(text_vector))

        if not vectors:
            raise QueryEmbeddingError("all image_text query embeddings failed")
        return QueryEmbeddingPlan(
            vectors=tuple(vectors),
            degraded=bool(reasons),
            degraded_reasons=tuple(reasons),
        )

    async def _embed_image_fallback(self, image_url: str) -> list[float] | None:
        try:
            result = await self._call(self._client.embed_image(image_url))
            return self._normalize(result)
        except Exception:
            logger.warning("separate image embedding fallback failed", exc_info=True)
            return None

    def _semantic_text_vectors(self, vector: list[float]) -> list[QueryVector]:
        return [
            self._query_vector(
                "asset_description",
                EmbeddingType.ASSET_DESCRIPTION,
                vector,
                self._settings.search_description_weight,
            ),
            self._query_vector(
                "subject_content",
                EmbeddingType.SUBJECT_CONTENT,
                vector,
                self._settings.search_subject_weight,
            ),
            self._query_vector(
                "visual_style",
                EmbeddingType.VISUAL_STYLE,
                vector,
                self._settings.search_visual_weight,
            ),
        ]

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

    @staticmethod
    def _query_vector(
        channel: str,
        embedding_type: EmbeddingType,
        vector: list[float],
        weight: float,
    ) -> QueryVector:
        return QueryVector(
            channel=channel,
            embedding_type=embedding_type,
            vector=vector,
            weight=weight,
        )
