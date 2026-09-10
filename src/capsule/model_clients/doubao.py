import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, ValidationError

from capsule.config import Settings
from capsule.enums import (
    AssetType,
    EmbeddingType,
    FeatureApplicability,
    FeatureSalience,
    FeatureStatus,
)
from capsule.features import (
    ACTIVE_EMBEDDING_TYPES,
    FEATURE_DIMENSION_SCOPES,
    embedding_type_supports_asset_type,
)
from capsule.model_clients.concurrency import AsyncCallPool
from capsule.model_clients.structured_output import responses_json_schema_format
from capsule.schemas import AssetFeatures, AssetUnderstanding, ClusterSummary, EmbeddingResult
from capsule.search.models import QueryEnhancement, SearchDimensionSuggestionResponse

ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)

_ASSET_UNDERSTANDING_RESPONSE_FORMAT = responses_json_schema_format(
    AssetUnderstanding,
    name="asset_understanding",
)
_ASSET_FEATURE_NAMES = frozenset(item.value for item in FEATURE_DIMENSION_SCOPES)


class _SearchQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: dict[EmbeddingType, str] = Field(min_length=1, max_length=4)
    weights: dict[EmbeddingType, StrictFloat | StrictInt] = Field(
        min_length=1,
        max_length=4,
    )


_EMBEDDING_TYPE_GUIDANCE: dict[EmbeddingType, str] = {
    EmbeddingType.NATIVE_MULTIMODAL: (
        "原始内容：聚焦原查询中可见或可读的主体、动作、场景、关系和内容约束，"
        "形成对素材本身的整体内容表达"
    ),
    EmbeddingType.SUBJECT_CONTENT: FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT],
    EmbeddingType.SCENE_THEME: FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME],
    EmbeddingType.VISUAL_PRESENTATION: FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_PRESENTATION],
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
                    "Authorization": (f"Bearer {settings.deepseek_api_key.get_secret_value()}"),
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
                        f"必须包含当前 Schema 指定的 {len(AssetFeatures.model_fields)} 个命名字段："
                        f"{', '.join(AssetFeatures.model_fields)}，绝不能使用数组。"
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
            raise DoubaoResponseError("embedding_types must be non-empty and contain no duplicates")
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
                "绝不虚构原文没有的人物、物体、场景、颜色、风格或其他具体事实，"
                "不得编造补全。"
                "dimension_guidance 仅用于说明关注范围，不能把其中的类别示例或枚举词"
                "复制进 query；query 中的每一项具体语义都必须能在 query_text 中找到依据。"
                "维度偏好和权重控制意图应体现在 weights 中，不应原样混入 queries；应自然"
                "保留实际检索语义以及否定、排除、范围等内容约束。native_multimodal 要"
                "围绕原始可见或可读内容及其关系组织整体表达。"
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
        if any(not math.isfinite(weight) or weight <= 0 for weight in parsed.weights.values()):
            raise DoubaoResponseError("query enhancer weights must be positive finite numbers")
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
                '{"embedding_types":["native_multimodal","visual_presentation"],'
                '"weights":{"native_multimodal":0.3,"visual_presentation":0.7}}。'
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
            for embedding_type in ACTIVE_EMBEDDING_TYPES
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
    return {
        "role": "system",
        "content": (
            "严格按照随请求提供的 JSON Schema 输出一个紧凑单行 JSON 对象，不要缩进、"
            "换行、Markdown、解释或代码围栏。description 表示当前维度的事实本身，"
            "evidence 表示支持该事实的可核验依据；两者必须分别填写，不能用 evidence 代替"
            "description。features 必须是对象且包含 Schema 指定的三个字段，不能输出数组。"
            "每个 item 完整包含 description、salience、status、evidence 和 "
            "ocr_confidence；subject_content 的每个 item 同时包含 subject。"
            "subject 是主体名称，description 是主体描述，不能互相代替；"
            "subject_content.salience 使用 0 到 1 的数字，其他 Feature 的 salience "
            "仍使用 high、medium 或 low；"
            "ocr_confidence 输出 null。"
        ),
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

    valid_applicability = {item.value for item in FeatureApplicability}
    valid_salience = {item.value for item in FeatureSalience}
    valid_statuses = {item.value for item in FeatureStatus}
    for feature in feature_values:
        if isinstance(feature, str):
            actions.add("feature_string_expanded")
            continue
        if not isinstance(feature, Mapping):
            continue
        if "items" not in feature:
            actions.add("legacy_feature_shape_expanded")
            continue
        if feature.get("applicability") not in valid_applicability:
            actions.add("feature_applicability_normalized")
        items = feature.get("items")
        if not isinstance(items, list):
            actions.add("feature_items_normalized")
            continue
        for item in items:
            if not isinstance(item, Mapping):
                actions.add("feature_items_normalized")
                continue
            raw_salience = item.get("salience")
            valid_relative_salience = (
                isinstance(raw_salience, (int, float))
                and not isinstance(raw_salience, bool)
                and 0.0 <= raw_salience <= 1.0
            )
            if raw_salience not in valid_salience and not valid_relative_salience:
                actions.add("feature_salience_normalized")
            if item.get("status") not in valid_statuses:
                actions.add("feature_status_normalized")
            evidence = item.get("evidence", [])
            if (
                isinstance(evidence, str)
                or evidence is None
                or not isinstance(evidence, list)
                or len(evidence) > 1
                or any(not isinstance(entry, str) or len(entry) > 80 for entry in evidence)
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
        if item_type == "audio_url":
            audio_url = item.get("audio_url")
            if not isinstance(audio_url, str):
                raise DoubaoResponseError("Responses audio content must contain a URL")
            converted.append({"type": "input_audio", "audio_url": audio_url})
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
