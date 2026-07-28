import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from capsule.config import Settings
from capsule.model_clients.concurrency import AsyncCallPool
from capsule.schemas import AssetUnderstanding, ClusterSummary, EmbeddingResult

ModelT = TypeVar("ModelT", bound=BaseModel)


class DoubaoConfigurationError(RuntimeError):
    pass


class DoubaoResponseError(RuntimeError):
    pass


class DoubaoClient:
    """Async client with independent concurrency pools per model workload."""

    def __init__(self, settings: Settings) -> None:
        if settings.ark_api_key is None:
            raise DoubaoConfigurationError("CAPSULE_ARK_API_KEY is required")

        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ark_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.ark_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            limits=httpx.Limits(
                max_connections=max(
                    32,
                    settings.understanding_concurrency + settings.embedding_concurrency,
                ),
                max_keepalive_connections=24,
            ),
        )
        self.understanding_pool = AsyncCallPool(
            name="understanding",
            concurrency=settings.understanding_concurrency,
            max_attempts=settings.model_max_retries,
        )
        self.embedding_pool = AsyncCallPool(
            name="embedding",
            concurrency=settings.embedding_concurrency,
            max_attempts=settings.model_max_retries,
        )
        self.capsule_pool = AsyncCallPool(
            name="capsule",
            concurrency=settings.capsule_concurrency,
            max_attempts=settings.model_max_retries,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DoubaoClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def understand_asset(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> AssetUnderstanding:
        return await self._chat_json(
            messages=messages,
            output_type=AssetUnderstanding,
            pool=self.understanding_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
        )

    async def summarize_cluster(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ClusterSummary:
        return await self._chat_json(
            messages=messages,
            output_type=ClusterSummary,
            pool=self.capsule_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
        )

    async def embed_multimodal(
        self,
        input_items: Sequence[Mapping[str, Any]],
    ) -> EmbeddingResult:
        async def request() -> EmbeddingResult:
            response = await self._client.post(
                "/embeddings/multimodal",
                json={
                    "model": self._settings.embedding_model,
                    "encoding_format": "float",
                    "input": list(input_items),
                },
                timeout=self._settings.embedding_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            vector = _extract_embedding(payload)
            if len(vector) != self._settings.embedding_dimension:
                raise DoubaoResponseError(
                    "embedding dimension mismatch: "
                    f"expected {self._settings.embedding_dimension}, got {len(vector)}"
                )
            return EmbeddingResult(
                vector=vector,
                model=str(payload.get("model", self._settings.embedding_model)),
                usage=payload.get("usage") or {},
                request_id=response.headers.get("x-request-id") or payload.get("id"),
            )

        return await self.embedding_pool.run(request)

    async def _chat_json(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[ModelT],
        pool: AsyncCallPool,
        timeout_seconds: float,
    ) -> ModelT:
        async def request() -> ModelT:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._settings.understanding_model,
                    "messages": list(messages),
                    "response_format": {"type": "json_object"},
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = _extract_message_content(payload)
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError as exc:
                raise DoubaoResponseError("model response is not valid JSON") from exc
            return output_type.model_validate(decoded)

        return await pool.run(request)


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DoubaoResponseError("chat response does not contain message content") from exc
    if not isinstance(content, str):
        raise DoubaoResponseError("chat message content must be a JSON string")
    return content


def _extract_embedding(payload: Mapping[str, Any]) -> list[float]:
    try:
        raw = payload["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DoubaoResponseError("embedding response does not contain a vector") from exc

    while isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        raise DoubaoResponseError("embedding value must be a list")

    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise DoubaoResponseError("embedding contains a non-numeric value") from exc
