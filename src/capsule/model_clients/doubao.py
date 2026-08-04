import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.model_clients.concurrency import AsyncCallPool
from capsule.schemas import AssetUnderstanding, ClusterSummary, EmbeddingResult
from capsule.search.models import (
    DimensionQuery,
    ParsedQuery,
    QueryDimensionSource,
    RerankBatch,
    SearchRequest,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class _SearchParserDimension(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_type: EmbeddingType
    query: str = Field(min_length=1, max_length=8_000)
    weight: float = Field(gt=0, le=1)
    source: QueryDimensionSource = QueryDimensionSource.TEXT


class _SearchParserOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_queries: list[_SearchParserDimension] = Field(min_length=1, max_length=12)


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
                max_connections=settings.http_max_connections,
                max_keepalive_connections=min(
                    settings.http_max_keepalive_connections,
                    settings.http_max_connections,
                ),
            ),
        )
        self.asset_understanding_pool = AsyncCallPool(
            name="asset_understanding",
            concurrency=settings.understanding_concurrency,
            max_attempts=settings.model_max_retries,
        )
        self.search_understanding_pool = AsyncCallPool(
            name="search_understanding",
            concurrency=settings.search_understanding_concurrency,
            max_attempts=settings.model_max_retries,
        )
        self.native_embedding_pool = AsyncCallPool(
            name="native_embedding",
            concurrency=settings.native_embedding_concurrency,
            max_attempts=settings.model_max_retries,
        )
        self.embedding_pool = AsyncCallPool(
            name="text_embedding",
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
        constrained_messages = [
            _asset_understanding_schema_message(),
            *messages,
        ]
        try:
            return await self._responses_json(
                messages=constrained_messages,
                output_type=AssetUnderstanding,
                pool=self.asset_understanding_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
            )
        except (DoubaoResponseError, ValidationError) as exc:
            correction = {
                "role": "user",
                "content": (
                    "上一份输出未通过 AssetUnderstanding 结构校验。请根据原始素材重新输出，"
                    "不要解释或使用 Markdown。根节点和 features 都必须是 JSON 对象；features "
                    "必须包含指定的十个命名字段，绝不能使用数组。"
                    f"校验错误：{_validation_error_text(exc)}"
                ),
            }
            return await self._responses_json(
                messages=[*constrained_messages, correction],
                output_type=AssetUnderstanding,
                pool=self.asset_understanding_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
            )

    async def summarize_cluster(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ClusterSummary:
        try:
            return await self._responses_json(
                messages=messages,
                output_type=ClusterSummary,
                pool=self.capsule_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
            )
        except ValidationError as exc:
            # Responses can be valid JSON but still violate the persisted Capsule
            # contract (most often a description shorter than 50 Chinese chars).
            # Retry once with the original representatives intact and explicit errors.
            validation_errors = json.dumps(
                exc.errors(include_url=False),
                ensure_ascii=False,
                default=str,
            )
            correction = {
                "role": "user",
                "content": (
                    "上一份输出未通过结构校验。请仅基于前述代表资产重新输出完整合法 JSON，"
                    "不要解释或使用 Markdown。description 必须是 50 到 150 个中文字符；"
                    "keywords 必须为 3 到 8 项；internal_variance 只能为 low、medium 或 high。"
                    f"校验错误：{validation_errors}"
                ),
            }
            return await self._responses_json(
                messages=[*messages, correction],
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
        """Parse a text/image query and normalize user-intent route weights."""
        system = {
            "role": "system",
            "content": (
                "你是多模态素材检索 Query Parser。只输出 JSON，根节点只能包含 "
                "dimension_queries。dimension_queries 必须且只能包含"
                "用户指令 required_embedding_types 中列出的维度，每项必须含"
                " embedding_type、query、weight、source。"
                "embedding_type 只能从 native_multimodal、asset_description、"
                "subject_content、scene_theme、visual_style、color_composition、"
                "mood_atmosphere、character_state_or_psychology、asset_usage、"
                "target_audience、provenance、rights_version_authorship 中选择且不得重复。"
                "source 只能是 text/image/joint。weight 必须反映 query_text 对已选维度的"
                "明确倾向且总和为 1；"
                "出现“重点、主要、更看重、其次”等表达时拉开权重，没有倾向时等权。"
                "禁止通过权重增加或删除 required_embedding_types。"
            ),
        }
        instruction = (
            f"query_type={request.query_type.value}; "
            f"precision_mode={request.precision_mode}; "
            "required_embedding_types="
            f"{','.join(item.value for item in request.embedding_types)}; "
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
        parsed = await self._responses_json(
            messages=[system, {"role": "user", "content": content}],
            output_type=_SearchParserOutput,
            pool=self.search_understanding_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
            max_output_tokens=self._settings.search_parser_max_output_tokens,
            model=self._settings.search_parser_model,
        )
        returned_types = [item.embedding_type for item in parsed.dimension_queries]
        if len(returned_types) != len(set(returned_types)) or set(returned_types) != set(
            request.embedding_types
        ):
            raise DoubaoResponseError(
                "search parser dimensions do not match required_embedding_types"
            )
        total_weight = sum(item.weight for item in parsed.dimension_queries)
        return ParsedQuery(
            dimension_queries=[
                DimensionQuery(
                    embedding_type=item.embedding_type,
                    query=item.query,
                    weight=item.weight / total_weight,
                    source=item.source,
                )
                for item in parsed.dimension_queries
            ],
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
                    "dimensions": self._settings.embedding_dimension,
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

        pool = (
            self.native_embedding_pool
            if _contains_visual_embedding_input(input_items)
            else self.embedding_pool
        )
        return await pool.run(request)

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

    async def _responses_json(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[ModelT],
        pool: AsyncCallPool,
        timeout_seconds: float,
        max_output_tokens: int | None = None,
        model: str | None = None,
    ) -> ModelT:
        """Call Ark Responses API for the Lite model with thinking disabled."""

        response_input = _responses_input(messages)

        async def request() -> ModelT:
            response = await self._client.post(
                "/responses",
                json={
                    "model": model or self._settings.understanding_model,
                    "input": response_input,
                    "thinking": {"type": "disabled"},
                    "max_output_tokens": (
                        max_output_tokens
                        if max_output_tokens is not None
                        else self._settings.understanding_max_output_tokens
                    ),
                    "text": {"format": {"type": "json_object"}},
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            try:
                decoded = json.loads(_extract_response_output_text(response.json()))
            except json.JSONDecodeError as exc:
                raise DoubaoResponseError("model response is not valid JSON") from exc
            return output_type.model_validate(decoded)

        return await pool.run(request)


def _asset_understanding_schema_message() -> dict[str, str]:
    example = json.dumps(
        _asset_understanding_json_example(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "role": "system",
        "content": (
            "以下输出结构约束优先于其他格式描述。只返回一个符合 JSON Schema 的 JSON 对象，"
            "不要返回 Markdown、解释或代码围栏。features 必须是对象，不能是数组；它必须包含"
            "十个命名 Feature 字段。每个 Feature 必须是包含 value、status、confidence、evidence "
            "的对象；value 最多五个关键词，evidence 最多一条，无证据时使用 null 和空数组。"
            "unknown 表示维度适用但证据不足，not_applicable 表示当前 Asset 不适用该维度；"
            "这两种状态的 value 必须为 null。人物状态维度在没有清晰可见或明确描述的人物、"
            "拟人角色时必须使用 not_applicable，禁止用场景、物体或怪物状态代替人物状态。"
            "asset_name 和 asset_description 必须以素材本身为主体；文件名、相对路径、"
            "目录层级、标题和关联文字中与素材一致的有效语义必须自然融入描述，但不得"
            "机械复述文件名、扩展名、目录、路径、来源路径或“位于某文件夹”等元数据措辞。"
            "路径与素材冲突时以素材为准，纯编号、序号或通用文件名必须忽略。"
            "唯一例外是 asset_usage：它除了通用字段外还必须返回 description 和 source_path。"
            "source_path 必须逐字复制输入 metadata.context.source_path；description 必须明确"
            "说明该完整相对路径及其对应用途。目录语义能确认用途时 status 使用 metadata，"
            "value 只写用途语义，不得写绝对路径。"
            "下面的手工示例只说明结构，禁止照抄；实际值必须根据输入素材重新判断。"
            f"JSON 结构示例：{example}"
        ),
    }


def _asset_understanding_json_example() -> dict[str, object]:
    observed: dict[str, object] = {
        "value": "关键词一；关键词二",
        "status": "observed",
        "confidence": 0.9,
        "evidence": ["输入中可核验的简短证据"],
    }
    unknown: dict[str, object] = {
        "value": None,
        "status": "unknown",
        "confidence": 0.0,
        "evidence": [],
    }
    not_applicable: dict[str, object] = {
        "value": None,
        "status": "not_applicable",
        "confidence": 1.0,
        "evidence": [],
    }
    asset_usage: dict[str, object] = {
        "value": "海报制作",
        "status": "metadata",
        "confidence": 0.95,
        "evidence": ["相对文件路径：海报/素材/example.png"],
        "description": (
            "该素材对应相对文件路径「海报/素材/example.png」，"
            "所属目录为「海报/素材」，路径语义表明其用于海报制作。"
        ),
        "source_path": "海报/素材/example.png",
    }
    return {
        "asset_name": "基于素材生成的简洁名称",
        "asset_description": "基于素材生成的客观完整描述",
        "features": {
            "subject_content": observed,
            "scene_theme": observed,
            "visual_style": observed,
            "color_composition": observed,
            "mood_atmosphere": observed,
            "character_state_or_psychology": not_applicable,
            "asset_usage": asset_usage,
            "target_audience": unknown,
            "provenance": unknown,
            "rights_version_authorship": unknown,
        },
    }


def _validation_error_text(exc: DoubaoResponseError | ValidationError) -> str:
    if isinstance(exc, ValidationError):
        return json.dumps(exc.errors(include_url=False), ensure_ascii=False, default=str)
    return str(exc)


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DoubaoResponseError("chat response does not contain message content") from exc
    if not isinstance(content, str):
        raise DoubaoResponseError("chat message content must be a JSON string")
    return content


def _responses_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    response_input: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, str):
            rendered_content = [{"type": "input_text", "text": content}]
        elif isinstance(content, Mapping):
            rendered_content = _responses_content([content])
        elif isinstance(content, list):
            rendered_content = _responses_content(content)
        else:
            raise DoubaoResponseError("Responses input content must be text or JSON data")
        response_input.append({"role": role, "content": rendered_content})
    return response_input


def _responses_content(items: Sequence[object]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise DoubaoResponseError("Responses content items must be JSON objects")
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str):
                raise DoubaoResponseError("Responses text content must contain text")
            converted.append({"type": "input_text", "text": text})
            continue
        if item_type == "image_url":
            image = item.get("image_url")
            if isinstance(image, Mapping):
                image_url = image.get("url")
                detail = image.get("detail")
            else:
                image_url = image
                detail = None
            if not isinstance(image_url, str):
                raise DoubaoResponseError("Responses image content must contain a URL")
            converted_image: dict[str, Any] = {
                "type": "input_image",
                "image_url": image_url,
            }
            if isinstance(detail, str):
                converted_image["detail"] = detail
            converted.append(converted_image)
            continue
        raise DoubaoResponseError(f"unsupported Responses content type: {item_type}")
    return converted


def _contains_visual_embedding_input(
    input_items: Sequence[Mapping[str, Any]],
) -> bool:
    return any(item.get("type") in {"image_url", "video_url"} for item in input_items)


def _extract_response_output_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        raise DoubaoResponseError("Responses payload does not contain output text")
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                return text
    raise DoubaoResponseError("Responses payload does not contain output text")


def _extract_embedding(payload: Mapping[str, Any]) -> list[float]:
    try:
        data = payload["data"]
        if isinstance(data, list):
            raw = data[0]["embedding"]
        elif isinstance(data, Mapping):
            raw = data["embedding"]
        else:
            raise TypeError("embedding data must be a list or object")
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
