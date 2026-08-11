import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from capsule.config import Settings
from capsule.enums import AssetType, EmbeddingType, FeatureStatus
from capsule.features import (
    FEATURE_DIMENSION_SCOPES,
    embedding_type_supports_asset_type,
)
from capsule.model_clients.concurrency import AsyncCallPool
from capsule.model_clients.structured_output import responses_json_schema_format
from capsule.schemas import AssetUnderstanding, ClusterSummary, EmbeddingResult
from capsule.search.models import QueryEnhancement, SearchDimensionSuggestionResponse

ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)

_ASSET_UNDERSTANDING_RESPONSE_FORMAT = responses_json_schema_format(
    AssetUnderstanding,
    name="asset_understanding",
    strip_annotations=True,
)
_ASSET_FEATURE_NAMES = frozenset(item.value for item in FEATURE_DIMENSION_SCOPES)


class _SearchQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: dict[EmbeddingType, str] = Field(min_length=1, max_length=12)
    weights: dict[EmbeddingType, StrictFloat | StrictInt] = Field(
        min_length=1,
        max_length=12,
    )


_EMBEDDING_TYPE_GUIDANCE: dict[EmbeddingType, str] = {
    EmbeddingType.NATIVE_MULTIMODAL: (
        "原始内容：聚焦原查询中可见或可读的主体、动作、场景、关系和内容约束，"
        "形成对素材本身的整体内容表达"
    ),
    EmbeddingType.ASSET_DESCRIPTION: (
        "素材完整描述：基于原查询已有信息，组织成对目标素材整体内容的客观完整描述"
    ),
    EmbeddingType.SUBJECT_CONTENT: FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT],
    EmbeddingType.SCENE_THEME: FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME],
    EmbeddingType.VISUAL_STYLE: FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_STYLE],
    EmbeddingType.COLOR_COMPOSITION: FEATURE_DIMENSION_SCOPES[EmbeddingType.COLOR_COMPOSITION],
    EmbeddingType.MOOD_ATMOSPHERE: FEATURE_DIMENSION_SCOPES[EmbeddingType.MOOD_ATMOSPHERE],
    EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY: FEATURE_DIMENSION_SCOPES[
        EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY
    ],
    EmbeddingType.ASSET_USAGE: FEATURE_DIMENSION_SCOPES[EmbeddingType.ASSET_USAGE],
    EmbeddingType.TARGET_AUDIENCE: FEATURE_DIMENSION_SCOPES[EmbeddingType.TARGET_AUDIENCE],
    EmbeddingType.PROVENANCE: FEATURE_DIMENSION_SCOPES[EmbeddingType.PROVENANCE],
    EmbeddingType.RIGHTS_VERSION_AUTHORSHIP: FEATURE_DIMENSION_SCOPES[
        EmbeddingType.RIGHTS_VERSION_AUTHORSHIP
    ],
}


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
        self._deepseek_client = (
            httpx.AsyncClient(
                base_url=settings.deepseek_base_url.rstrip("/"),
                headers={
                    "Authorization": (
                        f"Bearer {settings.deepseek_api_key.get_secret_value()}"
                    ),
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
            if settings.deepseek_api_key is not None
            else None
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
        if self._deepseek_client is not None:
            await self._deepseek_client.aclose()

    async def __aenter__(self) -> "DoubaoClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def understand_asset(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        asset_id: str | None = None,
    ) -> AssetUnderstanding:
        constrained_messages = [
            _asset_understanding_schema_message(),
            *messages,
        ]
        trace_asset_id = asset_id or "unknown"
        async with self.asset_understanding_pool.reserve() as slot:
            first_started = time.perf_counter()
            try:
                result, decoded = await slot.run(
                    lambda: self._responses_json_request(
                        messages=constrained_messages,
                        output_type=AssetUnderstanding,
                        timeout_seconds=self._settings.understanding_timeout_seconds,
                        response_format=_ASSET_UNDERSTANDING_RESPONSE_FORMAT,
                    )
                )
                normalization_attempt = "first"
                normalization_request_ms = (time.perf_counter() - first_started) * 1000
            except (DoubaoResponseError, ValidationError) as exc:
                first_attempt_ms = (time.perf_counter() - first_started) * 1000
                validation_error = _validation_error_text(exc)[:2000]
                logger.warning(
                    "asset understanding requires model repair asset_id=%s "
                    "error_type=%s first_attempt_ms=%.1f validation_error=%s",
                    trace_asset_id,
                    type(exc).__name__,
                    first_attempt_ms,
                    validation_error,
                )
                correction = {
                    "role": "user",
                    "content": (
                        "上一份输出未通过 AssetUnderstanding 结构校验。请根据原始素材重新输出，"
                        "不要解释或使用 Markdown。根节点和 features 都必须是 JSON 对象；features "
                        "必须包含指定的十个命名字段，绝不能使用数组。"
                        f"校验错误：{validation_error}"
                    ),
                }
                repair_started = time.perf_counter()
                try:
                    result, decoded = await slot.run(
                        lambda: self._responses_json_request(
                            messages=[*constrained_messages, correction],
                            output_type=AssetUnderstanding,
                            timeout_seconds=self._settings.understanding_timeout_seconds,
                            response_format=_ASSET_UNDERSTANDING_RESPONSE_FORMAT,
                        )
                    )
                except Exception as repair_exc:
                    repair_ms = (time.perf_counter() - repair_started) * 1000
                    logger.error(
                        "asset understanding model repair failed asset_id=%s "
                        "error_type=%s first_attempt_ms=%.1f repair_ms=%.1f error=%s",
                        trace_asset_id,
                        type(repair_exc).__name__,
                        first_attempt_ms,
                        repair_ms,
                        (str(repair_exc) or type(repair_exc).__name__)[:2000],
                    )
                    raise
                repair_ms = (time.perf_counter() - repair_started) * 1000
                logger.info(
                    "asset understanding repaired asset_id=%s repair_method=model_retry "
                    "first_attempt_ms=%.1f repair_ms=%.1f",
                    trace_asset_id,
                    first_attempt_ms,
                    repair_ms,
                )
                normalization_attempt = "repair"
                normalization_request_ms = repair_ms
            _log_asset_understanding_normalization(
                asset_id=trace_asset_id,
                decoded=decoded,
                request_attempt=normalization_attempt,
                request_ms=normalization_request_ms,
            )
            return result

    async def summarize_cluster(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> ClusterSummary:
        try:
            return await self._deepseek_json(
                messages=messages,
                output_type=ClusterSummary,
                pool=self.capsule_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
                max_output_tokens=self._settings.understanding_max_output_tokens,
                model=self._settings.search_query_model,
            )
        except ValidationError as exc:
            # Text-model responses can be valid JSON but still violate the Capsule contract.
            # Retry once with the complete cluster text evidence intact and explicit errors.
            validation_errors = json.dumps(
                exc.errors(include_url=False),
                ensure_ascii=False,
                default=str,
            )
            correction = {
                "role": "user",
                "content": (
                    "上一份输出未通过结构校验。请仅基于前述全部簇内资产文本重新输出完整合法 JSON，"
                    "不要解释或使用 Markdown。description 必须是 30 到 80 个中文字符；"
                    "common_features 必须有 1 到 3 项；不要输出 keywords；"
                    "internal_variance 只能为 low、medium 或 high。"
                    f"校验错误：{validation_errors}"
                ),
            }
            return await self._deepseek_json(
                messages=[*messages, correction],
                output_type=ClusterSummary,
                pool=self.capsule_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
                max_output_tokens=self._settings.understanding_max_output_tokens,
                model=self._settings.search_query_model,
            )

    async def enhance_search_query(
        self,
        *,
        query_text: str,
        embedding_types: Sequence[EmbeddingType],
    ) -> QueryEnhancement:
        """Enhance selected text routes and resolve normalized route weights."""
        requested_types = list(embedding_types)
        if not requested_types or len(requested_types) != len(set(requested_types)):
            raise DoubaoResponseError(
                "embedding_types must be non-empty and contain no duplicates"
            )
        dimension_guidance = {
            item.value: _EMBEDDING_TYPE_GUIDANCE[item] for item in requested_types
        }
        system = {
            "role": "system",
            "content": (
                "你是素材检索维度 Query Enhancer。只输出 JSON，根节点必须且只能包含"
                " queries 和 weights。queries、weights 都必须是对象，二者的键必须且只能"
                "是 required_embedding_types 列出的全部维度，不能增加、删除或重复维度。"
                "queries 的每个值必须是非空字符串：根据 dimension_guidance 将 query_text"
                "重组为适合该维度向量检索的查询，以目标维度为中心提高该维度信息密度。"
                "其他信息由你判断其是否有助于理解目标维度；只保留必要上下文，允许保留"
                "能说明目标维度的跨维度关联，不要机械地按词或维度删除。原文没有直接"
                "说明目标维度时，也要基于原查询做保守的维度化表达，不能因缺少直接线索"
                "而输出空查询。"
                "绝不虚构原文没有的人物、物体、场景、颜色、风格、情绪、用途、来源、"
                "作者、版权或其他具体事实，不得编造补全。"
                "dimension_guidance 仅用于说明关注范围，不能把其中的类别示例或枚举词"
                "复制进 query；query 中的每一项具体语义都必须能在 query_text 中找到依据。"
                "维度偏好和权重控制意图应体现在 weights 中，不应原样混入 queries；应自然"
                "保留实际检索语义以及否定、排除、范围等内容约束。native_multimodal 要"
                "围绕原始可见或可读内容及其关系组织整体表达；asset_description 要形成"
                "完整客观描述，但都只能使用原查询已有信息。"
                "weights 的每个值必须是大于 0 的有限数字，总和应为 1。把 query_text 当作"
                "普通用户对目标素材的自然描述，不要求用户说出系统维度名；根据表达中各类"
                "信息的相对关注程度大致分配权重即可，不追求过度精确。看不出明显倾向时"
                "使用等权。不要输出 source、embedding_type 列表、解释或 Markdown。"
            ),
        }
        instruction = json.dumps(
            {
                "required_embedding_types": [item.value for item in requested_types],
                "dimension_guidance": dimension_guidance,
                "query_text": query_text,
            },
            ensure_ascii=False,
        )
        try:
            parsed = await self._deepseek_json(
                messages=[system, {"role": "user", "content": instruction}],
                output_type=_SearchQueryOutput,
                pool=self.search_understanding_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
                max_output_tokens=self._settings.search_query_max_output_tokens,
                model=self._settings.search_query_model,
            )
        except ValidationError as exc:
            raise DoubaoResponseError("query enhancer output is invalid") from exc

        requested_type_set = set(requested_types)
        if set(parsed.queries) != requested_type_set:
            raise DoubaoResponseError(
                "query enhancer query dimensions do not match required_embedding_types"
            )
        queries = {
            embedding_type: parsed.queries[embedding_type].strip()
            for embedding_type in requested_types
        }
        if any(not query for query in queries.values()):
            raise DoubaoResponseError("query enhancer queries must be non-empty")
        if set(parsed.weights) != requested_type_set:
            raise DoubaoResponseError(
                "query enhancer weight dimensions do not match required_embedding_types"
            )
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in parsed.weights.values()
        ):
            raise DoubaoResponseError(
                "query enhancer weights must be positive finite numbers"
            )
        total_weight = sum(parsed.weights.values())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise DoubaoResponseError("query enhancer weight total must be positive")
        return QueryEnhancement(
            queries=queries,
            weights={
                embedding_type: parsed.weights[embedding_type] / total_weight
                for embedding_type in requested_types
            },
        )

    async def select_search_dimensions(
        self,
        *,
        query_text: str,
        asset_types: Sequence[AssetType],
    ) -> SearchDimensionSuggestionResponse:
        """Select up to four dimensions and resolve explicit query preferences."""
        system = {
            "role": "system",
            "content": (
                "你是素材检索维度选择器。根据用户查询、目标素材类型以及全部候选维度，"
                "选择最有助于召回目标素材的最小维度集合。只输出 JSON，根节点必须且只能"
                "包含 embedding_types 和 weights，例如 "
                '{"embedding_types":["native_multimodal","visual_style"],'
                '"weights":{"native_multimodal":0.3,"visual_style":0.7}}。'
                "必须选择 1 到 4 个不同维度，只能使用候选维度中的 embedding_type，且所选"
                "维度必须支持至少一种目标素材类型。不要为了凑数增加维度，不要输出理由、"
                "查询改写或 Markdown。把 query_text 当作普通用户向素材管理员描述想找的"
                "东西，不要求用户知道系统维度名；大致判断哪些方面更影响检索并映射到内部"
                "维度，不要过度分析细微措辞。weights 的键必须与 embedding_types 完全一致，"
                "每个值必须大于 0 且总和为 1；按相对关注程度给出近似权重，看不出明显差异"
                "时使用等权。"
            ),
        }
        candidates = [
            {
                "embedding_type": embedding_type.value,
                "description": _EMBEDDING_TYPE_GUIDANCE[embedding_type],
                "supported_asset_types": [
                    asset_type.value
                    for asset_type in AssetType
                    if embedding_type_supports_asset_type(
                        embedding_type=embedding_type,
                        asset_type=asset_type,
                    )
                ],
            }
            for embedding_type in EmbeddingType
        ]
        parsed = await self._deepseek_json(
            messages=[
                system,
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query_text": query_text,
                            "target_asset_types": [item.value for item in asset_types],
                            "candidate_dimensions": candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            output_type=SearchDimensionSuggestionResponse,
            pool=self.search_understanding_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
            max_output_tokens=min(
                self._settings.search_query_max_output_tokens,
                256,
            ),
            model=self._settings.search_query_model,
        )
        selected = parsed.embedding_types
        if len(selected) != len(set(selected)):
            raise DoubaoResponseError("dimension selector returned duplicate dimensions")
        if any(
            not any(
                embedding_type_supports_asset_type(
                    embedding_type=embedding_type,
                    asset_type=asset_type,
                )
                for asset_type in asset_types
            )
            for embedding_type in selected
        ):
            raise DoubaoResponseError(
                "dimension selector returned a dimension unsupported by target asset types"
            )
        return parsed

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

    async def _deepseek_json(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[ModelT],
        pool: AsyncCallPool,
        timeout_seconds: float,
        max_output_tokens: int,
        model: str,
    ) -> ModelT:
        """Call DeepSeek Chat Completions with JSON output and thinking disabled."""
        client = self._deepseek_client
        if client is None:
            raise DoubaoConfigurationError(
                "CAPSULE_DEEPSEEK_API_KEY is required for text-model calls"
            )

        async def request() -> ModelT:
            response = await client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": list(messages),
                    "thinking": {"type": "disabled"},
                    "max_tokens": max_output_tokens,
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

        async def request() -> ModelT:
            result, _ = await self._responses_json_request(
                messages=messages,
                output_type=output_type,
                timeout_seconds=timeout_seconds,
                max_output_tokens=max_output_tokens,
                model=model,
            )
            return result

        return await pool.run(request)

    async def _responses_json_request(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[ModelT],
        timeout_seconds: float,
        max_output_tokens: int | None = None,
        model: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> tuple[ModelT, Any]:
        """Execute one Responses JSON request without acquiring a call-pool slot."""

        response = await self._client.post(
            "/responses",
            json={
                "model": model or self._settings.understanding_model,
                "input": _responses_input(messages),
                "thinking": {"type": "disabled"},
                "max_output_tokens": (
                    max_output_tokens
                    if max_output_tokens is not None
                    else self._settings.understanding_max_output_tokens
                ),
                "text": {
                    "format": dict(response_format or {"type": "json_object"}),
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        try:
            decoded = json.loads(_extract_response_output_text(response.json()))
        except json.JSONDecodeError as exc:
            raise DoubaoResponseError("model response is not valid JSON") from exc
        return output_type.model_validate(decoded), decoded


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
            "的对象；value 最多五条短语，按表现力和区分度从高到低排列，使用中文分号连接。"
            "短语结构由当前维度的语义决定。描述局部属性且需要明确归属时，可以使用"
            "“具体对象 + 维度事实”，例如 color_composition 写“桌子 红色；星空 深蓝”。"
            "scene_theme 在素材具有可辨识的整体环境、时间、事件或叙事情境时适用；只有角色"
            "三视图、产品白底陈列或孤立元素展示时使用 not_applicable。"
            "mood_atmosphere 依据画面或文本中可核验的光线、色彩、空间、天气、动作、声音和"
            "叙事表现概括整体氛围；人物内心、动机或性格只有在素材明确呈现时才可作为依据，"
            "线索不足时使用 unknown。"
            "character_state_or_psychology 在素材包含人物、拟人角色或文本明确描述人物状态时"
            "适用；中性陈列视角没有可区分的表情、姿态、身体或心理状态时使用 not_applicable。"
            "target_audience、provenance 和 rights_version_authorship 以素材或上下文中的明确"
            "信息为证据，证据不足时使用 unknown。所有事实来自素材中可见、可读或有可靠上下文"
            "证据的内容。对于用途、受众、来源、权利等素材级维度，可以使用“素材 + 维度事实”。"
            "evidence 最多一条，无证据时使用 null 和空数组。"
            "unknown 表示维度适用但证据不足，not_applicable 表示当前 Asset 不适用该维度；"
            "这两种状态的 value 必须为 null。"
            "asset_name 和 asset_description 必须以素材本身为主体；文件名、相对路径、"
            "目录层级、标题和关联文字中与素材一致的有效语义必须自然融入描述，但不得"
            "机械复述文件名、扩展名、目录、路径、来源路径或“位于某文件夹”等元数据措辞。"
            "路径与素材冲突时以素材为准，纯编号、序号或通用文件名必须忽略。"
            "唯一例外是 asset_usage：它除了通用字段外还必须返回 description 和 source_path。"
            "source_path 必须逐字复制输入 metadata.context.source_path；description 必须明确"
            "说明该完整相对路径及其对应用途。目录语义能确认用途时 status 使用 metadata，"
            "value 按上述格式写“素材 + 规范化用途语义”，不得写绝对路径。"
            "下面的手工示例只说明结构，禁止照抄；实际值必须根据输入素材重新判断。"
            f"JSON 结构示例：{example}"
        ),
    }


def _asset_understanding_json_example() -> dict[str, object]:
    def observed(value: str) -> dict[str, object]:
        return {
            "value": value,
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
    asset_usage: dict[str, object] = {
        "value": "素材 海报制作",
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
            "subject_content": observed("女孩手持雨伞；小狗跟随女孩"),
            "scene_theme": observed("雨夜城市街道中的同行；都市夜行叙事情境"),
            "visual_style": observed("写实摄影；电影化视觉语言；细腻雨雾质感"),
            "color_composition": observed("冷蓝主色与暖黄点光对比；平视中景构图"),
            "mood_atmosphere": observed("安静神秘；略带紧张感"),
            "character_state_or_psychology": observed("女孩神情专注；身体微微前倾"),
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


def _log_asset_understanding_normalization(
    *,
    asset_id: str,
    decoded: Any,
    request_attempt: str,
    request_ms: float,
) -> None:
    actions = _asset_understanding_normalization_actions(decoded)
    if not actions:
        return
    logger.info(
        "asset understanding normalized asset_id=%s repair_method=local_normalization "
        "request_attempt=%s request_ms=%.1f actions=%s",
        asset_id,
        request_attempt,
        request_ms,
        ",".join(actions),
    )


def _asset_understanding_normalization_actions(decoded: Any) -> list[str]:
    """Describe only deterministic changes made by AssetUnderstanding validators."""

    if not isinstance(decoded, Mapping):
        return []
    actions: set[str] = set()
    for field_name, max_length in (("asset_name", 40), ("asset_description", 500)):
        value = decoded.get(field_name)
        if isinstance(value, str) and (value != value.strip() or len(value) > max_length):
            actions.add("asset_text_bounded")

    features = decoded.get("features")
    feature_values: list[Any]
    if isinstance(features, list):
        actions.add("feature_array_to_object")
        feature_values = features
    elif isinstance(features, Mapping):
        missing = _ASSET_FEATURE_NAMES.difference(str(key) for key in features)
        if missing:
            actions.add("missing_features_filled")
        feature_values = list(features.values())
    else:
        return sorted(actions)

    valid_statuses = {status.value for status in FeatureStatus}
    for feature in feature_values:
        if isinstance(feature, str):
            actions.add("feature_string_expanded")
            continue
        if not isinstance(feature, Mapping):
            continue
        if "value" not in feature:
            actions.add("missing_feature_value_filled")
        raw_value = feature.get("value")
        if isinstance(raw_value, list):
            actions.add("feature_value_list_joined")
        if feature.get("status") not in valid_statuses:
            actions.add("feature_status_normalized")
        raw_confidence = feature.get("confidence", 0.0)
        try:
            numeric_confidence = float(raw_confidence)
        except (TypeError, ValueError, OverflowError):
            numeric_confidence = math.nan
        if (
            isinstance(raw_confidence, bool)
            or not isinstance(raw_confidence, (int, float))
            or not (
                math.isfinite(numeric_confidence) and 0.0 <= numeric_confidence <= 1.0
            )
        ):
            actions.add("feature_confidence_normalized")
        evidence = feature.get("evidence", [])
        if (
            isinstance(evidence, str)
            or evidence is None
            or not isinstance(evidence, list)
            or len(evidence) > 1
            or any(not isinstance(item, str) or len(item) > 80 for item in evidence)
        ):
            actions.add("feature_evidence_normalized")
    return sorted(actions)


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
