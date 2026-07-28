import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from capsule.config import Settings
from capsule.model_clients.concurrency import AsyncCallPool
from capsule.schemas import AssetUnderstanding, ClusterSummary, EmbeddingResult
from capsule.search.models import ParsedQuery, RerankBatch, SearchRequest

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

    async def parse_search_query(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> ParsedQuery:
        """Parse a text/image query into the weighted routes frozen in the POC spec."""
        system = {
            "role": "system",
            "content": (
                "你是多模态素材检索 Query Parser。只输出 JSON。返回 query_summary、"
                "dimension_queries、negative_terms、parser_mode。dimension_queries 每项必须含"
                " embedding_type、query、weight、source、constraint；weight 总和必须为 1，"
                "embedding_type 只能从 native_multimodal、asset_description、"
                "subject_content、scene_theme、visual_style、color_composition、"
                "mood_atmosphere、character_state_or_psychology、asset_usage、"
                "target_audience、provenance、rights_version_authorship 中选择且不得重复。"
                "source 只能是 text/image/joint，constraint 只能是 match/maintain/add/"
                "exclude/modify。图片精搜优先 native=.45、subject=.15、scene=.10、"
                "visual=.10、color=.10、mood=.10。图文检索要识别保持、更像、排除、"
                "只看、风格相似等约束，使用 Late Fusion，不要把约束丢掉。"
            ),
        }
        instruction = (
            f"query_type={request.query_type.value}; "
            f"precision_mode={request.precision_mode}; "
            f"query_text={request.query_text or ''}"
        )
        content: list[dict[str, object]] = [{"type": "text", "text": instruction}]
        if image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return await self._chat_json(
            messages=[system, {"role": "user", "content": content}],
            output_type=ParsedQuery,
            pool=self.understanding_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
        )

    async def rerank_search_results(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
        candidates: Sequence[Mapping[str, object]],
    ) -> RerankBatch:
        """Rerank at most 30 hydrated candidates and provide an explainable reason."""
        system = {
            "role": "system",
            "content": (
                "你是素材检索重排器。只输出 JSON："
                '{"items":[{"asset_id":"...","relevance_score":0.0,"reason":"..."}]}。'
                "必须只使用候选中的 asset_id，每个候选恰好一次；按相关度降序。"
                "同时遵守用户的保持、增加、修改和排除约束；排除项应给极低分。"
                "relevance_score 范围为 0 到 1，reason 简洁说明命中的内容、场景、"
                "风格、色彩或情绪。"
            ),
        }
        query = {
            "query_type": request.query_type.value,
            "query_text": request.query_text,
            "candidates": list(candidates)[:30],
        }
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(query, ensure_ascii=False),
            }
        ]
        if image_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        return await self._chat_json(
            messages=[system, {"role": "user", "content": content}],
            output_type=RerankBatch,
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

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed text in the same multimodal space used by indexed assets."""
        return await self.embed_multimodal([{"type": "text", "text": text}])

    async def embed_image(self, image_url: str) -> EmbeddingResult:
        """Embed one remotely accessible image."""
        return await self.embed_multimodal(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            ]
        )

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult:
        """Generate a joint image-and-text query embedding."""
        return await self.embed_multimodal(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {"type": "text", "text": text},
            ]
        )

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
