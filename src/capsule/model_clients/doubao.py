import asyncio
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
from capsule.relation_graph import (
    AssetEntityRelationResolution,
    CreatedGroupParent,
    CrossTreeRelationRequest,
    CrossTreeRelationResolution,
    EntityStructureOperationResolution,
    EntityStructureRequest,
    GroupEntityOperation,
    MergedEntityResolution,
    MergeEntityOperation,
    MetadataContentResolution,
    RelatedEntityPairRequest,
    RelatedEntityPairResolution,
    ReusedGroupParent,
    SeparateEntityOperation,
)
from capsule.schemas import AssetFeatures, AssetUnderstanding, ClusterSummary, EmbeddingResult
from capsule.search.models import QueryEnhancement, SearchDimensionSuggestionResponse

ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)

_ASSET_UNDERSTANDING_RESPONSE_FORMAT = responses_json_schema_format(
    AssetUnderstanding,
    name="asset_understanding",
)
_METADATA_CONTENT_RESPONSE_FORMAT = responses_json_schema_format(
    MetadataContentResolution,
    name="metadata_content_resolution",
)
_MERGED_ENTITY_RESPONSE_FORMAT = responses_json_schema_format(
    MergedEntityResolution,
    name="merged_entity_resolution",
)
_ASSET_ENTITY_RELATION_RESPONSE_FORMAT = responses_json_schema_format(
    AssetEntityRelationResolution,
    name="asset_entity_relation_resolution",
)
_ASSET_ENTITY_RELATION_BATCH_SIZE = 10
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


def _validate_entity_structure_round(
    request: EntityStructureRequest,
    resolution: EntityStructureOperationResolution,
) -> None:
    """Validate references and exact incoming-Entity coverage for one Agent round."""

    current_ids = {node.entity_id for node in request.current_graph.nodes}
    incoming_ids = {node.entity_id for node in request.incoming_entities}
    known_ids = current_ids | incoming_ids
    incoming_occurrences: dict[str, int] = {entity_id: 0 for entity_id in incoming_ids}

    def reference(entity_id: str, *, allow_current: bool = True) -> None:
        allowed_ids = known_ids if allow_current else incoming_ids
        if entity_id not in allowed_ids:
            raise DoubaoResponseError(
                f"entity structure operation references unknown entity_id: {entity_id}"
            )
        if entity_id in incoming_occurrences:
            incoming_occurrences[entity_id] += 1

    for operation in resolution.operations:
        if isinstance(operation, MergeEntityOperation):
            for entity_id in operation.source_entity_ids:
                reference(entity_id)
            if not incoming_ids.intersection(operation.source_entity_ids):
                raise DoubaoResponseError("merge must process at least one incoming Entity")
        elif isinstance(operation, SeparateEntityOperation):
            for entity_id in operation.entity_ids:
                reference(entity_id, allow_current=False)
        elif isinstance(operation, GroupEntityOperation):
            if isinstance(operation.parent, ReusedGroupParent):
                reference(operation.parent.parent_entity_id)
            elif not isinstance(operation.parent, CreatedGroupParent):  # pragma: no cover
                raise DoubaoResponseError("unsupported group parent mode")
            for child in operation.children:
                reference(child.child_entity_id)

    missing_ids = sorted(
        entity_id for entity_id, count in incoming_occurrences.items() if count == 0
    )
    repeated_ids = sorted(
        entity_id for entity_id, count in incoming_occurrences.items() if count > 1
    )
    if missing_ids:
        raise DoubaoResponseError(
            f"entity structure response omitted incoming entity_ids: {missing_ids}"
        )
    if repeated_ids:
        raise DoubaoResponseError(
            f"entity structure response processed incoming entity_ids repeatedly: {repeated_ids}"
        )


def _normalize_entity_structure_resolution(
    request: EntityStructureRequest,
    resolution: EntityStructureOperationResolution,
) -> EntityStructureOperationResolution:
    """Downgrade an impossible one-child virtual group without losing valid operations."""

    incoming_ids = {node.entity_id for node in request.incoming_entities}
    operations: list[Any] = []
    for operation in resolution.operations:
        if (
            isinstance(operation, GroupEntityOperation)
            and isinstance(operation.parent, CreatedGroupParent)
            and len(operation.children) < 2
        ):
            child_ids = [
                child.child_entity_id
                for child in operation.children
                if child.child_entity_id in incoming_ids
            ]
            if child_ids:
                operations.append(SeparateEntityOperation(type="separate", entity_ids=child_ids))
            continue
        operations.append(operation)
    return EntityStructureOperationResolution(operations=operations)


def _validate_related_entity_pairs(
    request: RelatedEntityPairRequest,
    resolution: RelatedEntityPairResolution,
) -> None:
    current_ids = {entity.entity_id for entity in request.current_entities}
    incoming_ids = {entity.entity_id for entity in request.incoming_entities}
    known_ids = current_ids | incoming_ids
    for pair in resolution.pairs:
        pair_ids = {pair.source_entity_id, pair.target_entity_id}
        if not pair_ids.issubset(known_ids):
            raise DoubaoResponseError("related Entity pair references an unknown Entity")
        if not pair_ids.intersection(incoming_ids):
            raise DoubaoResponseError("related Entity pair must include an incoming Entity")


def _validate_cross_tree_relations(
    request: CrossTreeRelationRequest,
    resolution: CrossTreeRelationResolution,
) -> None:
    left_ids = {entity.entity_id for entity in request.left_tree.nodes}
    right_ids = {entity.entity_id for entity in request.right_tree.nodes}
    for relation in resolution.relations:
        endpoints = {relation.source_entity_id, relation.target_entity_id}
        if not (
            endpoints.intersection(left_ids)
            and endpoints.intersection(right_ids)
            and endpoints.issubset(left_ids | right_ids)
        ):
            raise DoubaoResponseError("relation must connect one node from each Entity tree")


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

    async def resolve_metadata_content_entities(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        guidance: str | None = None,
    ) -> MetadataContentResolution:
        """Classify metadata/content relationships once per reusable metadata group."""

        system = {
            "role": "system",
            "content": guidance
            or (
                "合并元数据实体和内容实体。same_entity 表示元数据名称就是内容主体的身份；"
                "contains_content 表示元数据是容纳内容主体的场景或集合；其余为 related。"
                "结合组内多项内容整体判断，并为形成后的 Entity 生成 entity_semantic，描述"
                "该实体自身是谁或是什么。entity_semantic 保持实体名称对应的稳定语义层级，"
                "不因当前成员里出现的局部人物、物体或陈设而缩窄。用 build_entity 表示该"
                "元数据实体是否适合作为关系图谱节点。角色、地点、场景、组织、事件、道具等"
                "具有实际语义的实体设为 true；如果它只是用于存放或整理文件的通用容器，"
                "只呈现文件包含关系而不包含实际语义，设为 false，不参与关系构建。"
            ),
        }
        async with self.asset_understanding_pool.reserve() as slot:
            result, _ = await slot.run(
                lambda: self._responses_json_request(
                    messages=[
                        system,
                        {
                            "role": "user",
                            "content": json.dumps({"groups": list(groups)}, ensure_ascii=False),
                        },
                    ],
                    output_type=MetadataContentResolution,
                    timeout_seconds=self._settings.understanding_timeout_seconds,
                    response_format=_METADATA_CONTENT_RESPONSE_FORMAT,
                )
            )
        return result

    async def generate_asset_entity_relations(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> AssetEntityRelationResolution:
        """Judge relations between an Asset's internal subject and an Entity subject."""

        system = {
            "role": "system",
            "content": (
                "判断每组 asset 与 entity 是否存在直接关系。依据包括 asset.metadata、"
                "asset.content_description、asset.content_subject、entity.name 和 entity.semantic。"
                "综合元数据与内容理解二者的实际语义；元数据明确揭示直接关系时优先采用。"
                "目录路径按语义层级理解：越具体且与 Entity 语义对应的层级，越能支持直接关系；"
                "共同上层路径通常只说明共享背景或归属范围，不自然扩展为同级节点之间的关系。"
                "正例：路径中的人物与道具层级可支持该人物的道具 Entity；场景与具体区域层级可"
                "支持该区域 Entity；物件与部件层级可支持该部件 Entity。"
                "反例：共享同一人物上层路径，不表示服装素材属于武器 Entity；共享同一场景上层路径，"
                "不表示一个区域的素材属于另一个同级区域 Entity；纯项目或整理目录也不直接证明"
                "内容关系。自然生成 relation 和 description；不建立关系时用 description 简述"
                "关键差异。每项独立判断并原样返回 source_id、target_id，不输出 reason。"
                "输出 JSON 格式为"
                '{"relations":[{"source_id":"","target_id":"",'
                '"establishes_relation":true,"relation":"",'
                '"description":""}]}。'
            ),
        }
        batches = [
            list(candidates[offset : offset + _ASSET_ENTITY_RELATION_BATCH_SIZE])
            for offset in range(0, len(candidates), _ASSET_ENTITY_RELATION_BATCH_SIZE)
        ]
        if not batches:
            return AssetEntityRelationResolution()

        async def resolve_batch(
            batch: list[Mapping[str, Any]],
        ) -> AssetEntityRelationResolution:
            return await self._deepseek_json(
                messages=[
                    system,
                    {
                        "role": "user",
                        "content": json.dumps({"candidates": batch}, ensure_ascii=False),
                    },
                ],
                output_type=AssetEntityRelationResolution,
                pool=self.capsule_pool,
                timeout_seconds=self._settings.understanding_timeout_seconds,
                max_output_tokens=max(
                    self._settings.understanding_max_output_tokens,
                    4096,
                ),
                model=self._settings.search_query_model,
            )

        resolutions = await asyncio.gather(*(resolve_batch(batch) for batch in batches))
        return AssetEntityRelationResolution(
            relations=[relation for resolution in resolutions for relation in resolution.relations]
        )

    async def merge_entity_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> MergedEntityResolution:
        """Merge metadata and subject-cluster candidates before edge judgment."""

        system = {
            "role": "system",
            "content": (
                "输入是来自元数据和主体聚类的实体候选。判断哪些候选指向同一个实际实体，"
                "合并后生成统一的 name、semantic 和 candidate_ids。主体聚类只是候选发现结果，"
                "相似主题、风格或类别不代表同一实体；元数据候选也不天然正确。角色、地点、场景、"
                "组织、事件、道具等具有实际语义且能跨至少两个资产复用的实体可设 build_entity"
                "为 true。只是用于存放或整理文件的通用容器，或不具备可复用实际意义的候选设为"
                "false。不要在此判断 Asset 与 Entity 的最终关系。每个输入 candidate_id 在输出中"
                "出现一次，保留未与其他候选合并的有效实体候选，并在 reason 中说明合并或拒绝依据。"
                '只输出 JSON，格式为{"entities":[{"name":"","semantic":"",'
                '"candidate_ids":[""],"build_entity":true,"reason":""}]}。'
            ),
        }
        return await self._deepseek_json(
            messages=[
                system,
                {
                    "role": "user",
                    "content": json.dumps({"candidates": list(candidates)}, ensure_ascii=False),
                },
            ],
            output_type=MergedEntityResolution,
            pool=self.capsule_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
            max_output_tokens=max(
                self._settings.understanding_max_output_tokens,
                4096,
            ),
            model=self._settings.search_query_model,
        )

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: Mapping[str, Any],
        incoming_entities: Sequence[Mapping[str, Any]],
        workspace_tree: str,
    ) -> EntityStructureOperationResolution:
        """Generate one validated round of incremental Entity structure operations."""

        request = EntityStructureRequest.model_validate(
            {
                "workspace_tree": workspace_tree,
                "current_graph": dict(current_graph),
                "incoming_entities": list(incoming_entities),
            }
        )
        system = {
            "role": "system",
            "content": (
                "增量整理 Entity 结构。结合 Entity 的 name、semantic 与 workspace_tree 理解项目"
                "语义；目录树提供上下文和归属线索，current_graph 是已接受结构，incoming_entities"
                "是本轮节点。仅当两个节点的语义身份、抽象层级和用途都基本可互换时 merge；没有"
                "结构关系时 separate；共享明确的实际主体或上位语义、同时又应保留各自差异时"
                "group。共享主体不等于可合并：状态、时期、版本、组成部分或功能角色不同的节点"
                "保留区别；地点与其内部物件、人物与其装备等不同语义层级也不合并。group 可复用"
                "现有父节点或创建"
                "新的虚拟父节点，并自然描述每个 child 与 parent 的关系。"
                "正例：同一人物的不同状态、时期或版本可归入该人物；人物的服装、武器和动作设定可"
                "围绕该人物组织；同一地点的不同区域可归入该地点；同一装置的不同部件或工作状态可"
                "归入该装置。反例：仅画风相似、题材相近或同属宽泛类别，不足以合并或创建上层节点；"
                "“素材”“确认候选”“测试”等整理目录不是实际语义节点；名称相似但语义指向不同对象"
                "时保持分离。例子用于说明判断方式，不限定 Entity 类型或关系名称。"
                "每个 incoming entity_id 选择一次操作；新建父节点至少包含两个 child。只输出增量"
                "operations，不重写 current_graph，不输出 reason 或解释。新父节点使用 virtual:"
                "开头的临时 ID，后端会在下一轮继续处理。输出 JSON 格式为"
                '{"operations":['
                '{"type":"merge","source_entity_ids":[""],'
                '"canonical_entity_id":"","name":"","semantic":""},'
                '{"type":"separate","entity_ids":[""]},'
                '{"type":"group","parent":{"mode":"reuse",'
                '"parent_entity_id":""},"children":[{"child_entity_id":"",'
                '"relation":"","description":""}]},'
                '{"type":"group","parent":{"mode":"create",'
                '"temporary_parent_id":"virtual:group_name","name":"",'
                '"semantic":""},"children":[{"child_entity_id":"",'
                '"relation":"","description":""}]}'
                "]}。"
            ),
        }
        messages = [
            system,
            {
                "role": "user",
                "content": request.model_dump_json(),
            },
        ]
        for attempt in range(3):
            resolution: EntityStructureOperationResolution | None = None
            try:
                resolution = await self._deepseek_json(
                    messages=messages,
                    output_type=EntityStructureOperationResolution,
                    pool=self.capsule_pool,
                    timeout_seconds=self._settings.understanding_timeout_seconds,
                    max_output_tokens=max(
                        self._settings.understanding_max_output_tokens,
                        4096,
                    ),
                    model=self._settings.search_query_model,
                )
                resolution = _normalize_entity_structure_resolution(request, resolution)
                _validate_entity_structure_round(request, resolution)
            except (DoubaoResponseError, ValidationError) as exc:
                if attempt == 2:
                    raise
                if resolution is not None:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": resolution.model_dump_json(),
                        }
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"上一输出未通过结构校验：{exc}。请重新输出 operations，为每个 "
                            "incoming entity_id 选择一次 merge、group 或 separate。创建新的父"
                            "Entity 时放入至少两个具有共同实际主体的 child；只有一个 child 时"
                            "复用合适的现有父 Entity，或使用 separate。"
                        ),
                    }
                )
                continue
            assert resolution is not None
            return resolution
        raise AssertionError("unreachable")

    async def select_related_entity_pairs(
        self,
        *,
        current_entities: Sequence[Mapping[str, Any]],
        incoming_entities: Sequence[Mapping[str, Any]],
        workspace_tree: str,
    ) -> RelatedEntityPairResolution:
        """Recall top-level Entity-tree pairs that may contain a concrete relation."""

        request = RelatedEntityPairRequest.model_validate(
            {
                "workspace_tree": workspace_tree,
                "current_entities": list(current_entities),
                "incoming_entities": list(incoming_entities),
            }
        )
        resolution = await self._deepseek_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "从顶层 Entity 中筛选可能存在具体语义关系的 Entity 树对。结合名称、语义"
                        "和 workspace_tree 判断；这里只做候选召回，不生成关系，也不合并或分组。"
                        "人物与场景、人物与物件、事件与地点、地点与设施、组织与人物等存在可具体"
                        "描述关系时保留。仅同属项目、目录相邻、画风或题材相似、宽泛同类时不保留。"
                        "检查 incoming_entities 内部以及它们与 current_entities 的组合；每个输出对"
                        "至少包含一个 incoming Entity。只输出 JSON："
                        '{"pairs":[{"source_entity_id":"","target_entity_id":""}]}。'
                    ),
                },
                {"role": "user", "content": request.model_dump_json()},
            ],
            output_type=RelatedEntityPairResolution,
            pool=self.capsule_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
            max_output_tokens=max(self._settings.understanding_max_output_tokens, 4096),
            model=self._settings.search_query_model,
        )
        _validate_related_entity_pairs(request, resolution)
        return resolution

    async def generate_cross_tree_relations(
        self,
        *,
        left_tree: Mapping[str, Any],
        right_tree: Mapping[str, Any],
        workspace_tree: str,
    ) -> CrossTreeRelationResolution:
        """Create concrete semantic edges between arbitrary nodes of two Entity trees."""

        request = CrossTreeRelationRequest.model_validate(
            {
                "workspace_tree": workspace_tree,
                "left_tree": dict(left_tree),
                "right_tree": dict(right_tree),
            }
        )
        resolution = await self._deepseek_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "判断两棵 Entity 树之间的具体语义关系。关系端点可选择树中的任意节点，"
                        "不限定顶层节点；选择语义最准确的层级，并自然生成有方向的 relation 和"
                        "description。只建立由 Entity 语义或目录上下文明示的项目事实，不补充可能的"
                        "用途、出现场景、创作参考或因果关系。仅同属项目、目录相近、画风或题材相似"
                        "不构成关系；需要使用“可能”“可作为”“可供参考”“推测”或同时说明“无直接"
                        "关联”时，返回空数组。可建立多条互不重复的关系，关系名和描述使用与输入"
                        "一致的自然语言。不合并节点，不创建新节点。"
                        "只输出 JSON："
                        '{"relations":[{"source_entity_id":"","target_entity_id":"",'
                        '"relation":"","description":""}]}。'
                    ),
                },
                {"role": "user", "content": request.model_dump_json()},
            ],
            output_type=CrossTreeRelationResolution,
            pool=self.capsule_pool,
            timeout_seconds=self._settings.understanding_timeout_seconds,
            max_output_tokens=max(self._settings.understanding_max_output_tokens, 4096),
            model=self._settings.search_query_model,
        )
        _validate_cross_tree_relations(request, resolution)
        return resolution

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
