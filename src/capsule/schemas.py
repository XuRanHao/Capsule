from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from capsule.enums import (
    AssetIndexRole,
    AssetType,
    EmbeddingType,
    FeatureApplicability,
    FeatureSalience,
    FeatureStatus,
    ProcessingStatus,
)
from capsule.features import FEATURE_DIMENSION_SCOPES


class SourceContext(BaseModel):
    text: str
    relation_type: str
    text_block_index: int | None = None
    paragraph_id: str | None = None
    source_path: str | None = None
    document_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)


class DiscoveredFile(BaseModel):
    path: str
    relative_path: str
    extension: str
    size_bytes: int


class AssetDraft(BaseModel):
    asset_type: AssetType
    file_name: str
    index_role: AssetIndexRole = AssetIndexRole.STANDALONE
    hierarchy_key: str | None = None
    parent_hierarchy_key: str | None = None
    child_order: int | None = Field(default=None, ge=0)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    source_contexts: list[SourceContext] = Field(default_factory=list)
    raw_content: str | None = None
    file_info: dict[str, Any] = Field(default_factory=dict)
    derived_file_uri: str | None = None
    preview_uri: str | None = None
    transient_keyframe_jpegs: list[bytes] = Field(default_factory=list, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "AssetDraft":
        if self.parent_hierarchy_key is not None:
            if self.index_role == AssetIndexRole.STANDALONE:
                self.index_role = AssetIndexRole.CHILD
            elif self.index_role != AssetIndexRole.CHILD:
                raise ValueError("only child assets may reference a parent_hierarchy_key")
        elif self.index_role == AssetIndexRole.CHILD:
            raise ValueError("child assets require a parent_hierarchy_key")
        if self.index_role == AssetIndexRole.PARENT and not self.hierarchy_key:
            raise ValueError("parent assets require a hierarchy_key")
        if self.index_role == AssetIndexRole.CHILD and self.child_order is None:
            raise ValueError("child assets require a child_order")
        if self.index_role != AssetIndexRole.CHILD and self.child_order is not None:
            raise ValueError("child_order is only valid for child assets")
        return self


class AssetPlayback(BaseModel):
    """Browser playback contract for derived clips and legacy logical ranges."""

    mode: Literal["derived_clip", "source_range", "transcoded_stream"]
    url: str
    fallback_url: str | None = None
    mime_type: str
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    browser_compatible: bool


class AssetCreate(BaseModel):
    asset_id: str
    workspace_id: str
    source_file_id: str
    asset_type: AssetType
    file_name: str
    file_type: str
    asset_key: str
    index_role: AssetIndexRole = AssetIndexRole.STANDALONE
    child_order: int | None = Field(default=None, ge=0)
    # Stable intra-batch reference only; repositories resolve it to parent_asset_id.
    parent_asset_key: str | None = Field(default=None, exclude=True, repr=False)
    generation: int = Field(default=0, ge=0)
    content_hash: str
    asset_name: str | None = None
    asset_description: str | None = None
    asset_features: dict[str, Any] = Field(default_factory=dict)
    file_tree_context: list[str] = Field(default_factory=list)
    source_contexts: list[SourceContext] = Field(default_factory=list)
    file_info: dict[str, Any] = Field(default_factory=dict)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: str | None = None
    derived_file_uri: str | None = None
    preview_uri: str | None = None
    transient_keyframe_jpegs: list[bytes] = Field(default_factory=list, exclude=True, repr=False)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    feature_revision: int = 1
    embedding_revision: int = 1

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "AssetCreate":
        if self.index_role == AssetIndexRole.CHILD and not self.parent_asset_key:
            raise ValueError("child assets require a parent_asset_key")
        if self.index_role != AssetIndexRole.CHILD and self.parent_asset_key is not None:
            raise ValueError("only child assets may reference a parent_asset_key")
        if self.index_role == AssetIndexRole.CHILD and self.child_order is None:
            raise ValueError("child assets require a child_order")
        if self.index_role != AssetIndexRole.CHILD and self.child_order is not None:
            raise ValueError("child_order is only valid for child assets")
        return self


class AssetRecord(BaseModel):
    asset_id: str
    workspace_id: str
    source_file_id: str
    asset_type: AssetType
    file_name: str
    file_type: str
    index_role: AssetIndexRole = AssetIndexRole.STANDALONE
    parent_asset_id: str | None = None
    child_order: int | None = None
    asset_name: str | None = None
    asset_description: str | None = None
    asset_features: dict[str, Any] = Field(default_factory=dict)
    file_tree_context: list[str] = Field(default_factory=list)
    file_info: dict[str, Any] = Field(default_factory=dict)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: str | None = None
    derived_file_uri: str | None = None
    preview_uri: str | None = None
    playback: AssetPlayback | None = None
    processing_status: ProcessingStatus
    feature_revision: int = 1
    embedding_revision: int = 1
    created_at: datetime
    updated_at: datetime


class AssetEmbeddingState(BaseModel):
    embedding_type: str
    status: str
    model_name: str
    embedding_revision: int | None = None


class AssetSourceRecord(BaseModel):
    source_file_id: str
    original_file_name: str
    relative_path: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    processing_status: str
    error_message: str | None = None


class AssetViewRecord(BaseModel):
    asset_id: str
    workspace_id: str
    project_id: str
    source_file_id: str
    asset_type: AssetType
    file_name: str
    file_type: str
    index_role: AssetIndexRole = AssetIndexRole.STANDALONE
    parent_asset_id: str | None = None
    child_order: int | None = None
    asset_name: str | None = None
    asset_description: str | None = None
    asset_features: dict[str, Any] = Field(default_factory=dict)
    file_tree_context: list[str] = Field(default_factory=list)
    source_contexts: list[dict[str, Any]] = Field(default_factory=list)
    file_info: dict[str, Any] = Field(default_factory=dict)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: str | None = None
    processing_status: str
    feature_revision: int
    embedding_revision: int
    error_message: str | None = None
    preview_url: str | None = None
    content_url: str | None = None
    playback: AssetPlayback | None = None
    # API-only routing inputs. They are populated by repositories but never
    # serialized, keeping storage URIs out of public asset views.
    source_storage_uri: str | None = Field(default=None, exclude=True)
    derived_file_uri: str | None = Field(default=None, exclude=True)
    preview_uri: str | None = Field(default=None, exclude=True)
    source_file: AssetSourceRecord
    embeddings: list[AssetEmbeddingState] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AssetListResponse(BaseModel):
    items: list[AssetViewRecord] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class WorkspaceRecord(BaseModel):
    workspace_id: str
    name: str
    created_at: datetime
    updated_at: datetime


class WorkspaceListResponse(BaseModel):
    items: list[WorkspaceRecord] = Field(default_factory=list)


class WorkspaceDeleteResult(BaseModel):
    workspace_id: str
    workspace_deleted: bool
    assets_deleted: int
    source_files_deleted: int
    embeddings_deleted: int
    jobs_deleted: int
    vectors_deleted: int
    objects_deleted: int
    staging_paths_deleted: int
    cancelled_jobs: int = 0
    cleanup_warnings: list[str] = Field(default_factory=list)


class LibraryClearResult(BaseModel):
    workspaces_deleted: int
    assets_deleted: int
    source_files_deleted: int
    embeddings_deleted: int
    jobs_deleted: int
    vectors_deleted: int
    objects_deleted: int
    staging_paths_deleted: int
    cleanup_warnings: list[str] = Field(default_factory=list)


class StoredFileResult(BaseModel):
    source_file_id: str
    asset_ids: list[str] = Field(default_factory=list)
    indexable_asset_ids: list[str] = Field(default_factory=list)


class FeatureItem(BaseModel):
    """One independently attributable fact inside a Feature dimension."""

    description: str = Field(
        min_length=1,
        max_length=120,
        description="当前维度内的一条事实本身，不是证据说明",
    )
    salience: FeatureSalience = Field(
        default=FeatureSalience.MEDIUM,
        description="该事实在当前维度内的相对显著程度",
    )
    status: FeatureStatus = Field(
        default=FeatureStatus.INFERRED,
        description="事实的依据类型",
    )
    evidence: list[str] = Field(
        default_factory=list,
        max_length=1,
        description="支持 description 的一条可核验依据；不能用它代替 description",
    )
    ocr_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="模型始终输出 null；若使用本地 OCR，真实置信度由后端写入",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_output(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = {"description": value}
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        description = normalized.get("description")
        if isinstance(description, str):
            normalized["description"] = description.strip()[:120]

        valid_salience = {item.value for item in FeatureSalience}
        raw_salience = normalized.get("salience")
        uses_relative_salience = cls.model_fields["salience"].annotation is float
        if uses_relative_salience:
            if isinstance(raw_salience, str) and raw_salience in valid_salience:
                normalized["salience"] = {
                    FeatureSalience.HIGH.value: 1.0,
                    FeatureSalience.MEDIUM.value: 0.5,
                    FeatureSalience.LOW.value: 0.2,
                }[raw_salience]
        elif raw_salience not in valid_salience:
            normalized["salience"] = FeatureSalience.MEDIUM.value
        valid_statuses = {item.value for item in FeatureStatus}
        if normalized.get("status") not in valid_statuses:
            normalized["status"] = FeatureStatus.INFERRED.value

        evidence = normalized.get("evidence")
        if isinstance(evidence, str):
            normalized["evidence"] = [evidence.strip()[:80]] if evidence.strip() else []
        elif isinstance(evidence, list):
            normalized["evidence"] = [
                item.strip()[:80] for item in evidence if isinstance(item, str) and item.strip()
            ][:1]
        else:
            normalized["evidence"] = []

        raw_ocr_confidence = normalized.get("ocr_confidence")
        if isinstance(raw_ocr_confidence, bool):
            normalized["ocr_confidence"] = None
        elif raw_ocr_confidence is not None:
            try:
                parsed = float(raw_ocr_confidence)
            except (TypeError, ValueError, OverflowError):
                parsed = -1.0
            normalized["ocr_confidence"] = parsed if 0.0 <= parsed <= 1.0 else None
        return normalized


class SubjectFeatureItem(FeatureItem):
    """One named subject plus the facts that distinguish it."""

    subject: str = Field(
        min_length=1,
        max_length=60,
        description=(
            "可跨素材复用的主体名称；优先采用输入元数据中与素材内容一致的有效专名，"
            "否则使用可观察到的具体类别名"
        ),
    )
    description: str = Field(
        min_length=1,
        max_length=120,
        description="该主体的身份、类别、外观、动作或关系等可区分事实，不是主体名称的重复",
    )
    # This Pydantic specialization intentionally changes the wire type from the
    # categorical salience used by ordinary features to a relative numeric score.
    salience: float = Field(  # type: ignore[assignment]
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "该主体在当前 Asset 中相对于其他主体的重要程度，取 0 到 1；"
            "多个不同层次的实体可以同时具有较高数值"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_subject_output(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"subject": value, "description": value}
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        subject = next(
            (
                normalized.get(field_name)
                for field_name in ("subject", "entity_name", "name")
                if isinstance(normalized.get(field_name), str)
                and normalized[field_name].strip()
            ),
            None,
        )
        description = normalized.get("description")
        # Historical subject_content items stored only description. Keeping this
        # fallback makes those records readable while new structured output is
        # required to provide subject explicitly.
        if subject is None and isinstance(description, str) and description.strip():
            subject = description
        if isinstance(subject, str):
            normalized["subject"] = subject.strip()[:60]
        return normalized


class FeatureValue(BaseModel):
    """A dimension-level envelope with zero to five salience-ranked facts."""

    applicability: FeatureApplicability = Field(
        description=(
            "applicable 表示有可描述事实；unknown 表示维度适用但证据不足；"
            "not_applicable 表示该维度不适用于当前素材"
        )
    )
    items: list[FeatureItem] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "当前维度内 0 到 5 条互不重复的事实，按 salience 从 high 到 low 排列；"
            "unknown 或 not_applicable 时必须为空"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_output(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _legacy_feature_value(value=value, status=FeatureStatus.INFERRED.value)
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "items" not in normalized:
            raw_value = next(
                (
                    normalized[field]
                    for field in ("effective_value", "user_value", "model_value", "value")
                    if field in normalized
                ),
                None,
            )
            return _legacy_feature_value(
                value=raw_value,
                status=normalized.get("status"),
                evidence=normalized.get("evidence"),
            )

        items = normalized.get("items")
        normalized["items"] = items if isinstance(items, list) else []
        valid_applicability = {item.value for item in FeatureApplicability}
        if normalized.get("applicability") not in valid_applicability:
            normalized["applicability"] = (
                FeatureApplicability.APPLICABLE.value
                if normalized["items"]
                else FeatureApplicability.UNKNOWN.value
            )
        return normalized

    @model_validator(mode="after")
    def normalize_items(self) -> "FeatureValue":
        if self.applicability is not FeatureApplicability.APPLICABLE:
            self.items = []
            return self
        if not self.items:
            self.applicability = FeatureApplicability.UNKNOWN
            return self

        priority = {
            FeatureSalience.HIGH: 0,
            FeatureSalience.MEDIUM: 1,
            FeatureSalience.LOW: 2,
        }
        unique: dict[str, FeatureItem] = {}
        for item in self.items:
            key = item.description.casefold()
            unique.setdefault(key, item)
        self.items = sorted(
            unique.values(),
            key=lambda item: (
                priority[item.salience]
                if isinstance(item.salience, FeatureSalience)
                else -float(item.salience)
            ),
        )[:5]
        return self

    @property
    def value(self) -> str | None:
        """Legacy read-only projection for internal callers during data migration."""

        if self.applicability is not FeatureApplicability.APPLICABLE:
            return None
        return "；".join(item.description for item in self.items) or None


class SubjectFeatureValue(FeatureValue):
    """Subject dimension whose items preserve names separately from descriptions."""

    # Pydantic supports narrowing a model field in a subclass; list invariance
    # means the static checker needs this explicit acknowledgement.
    items: list[SubjectFeatureItem] = Field(  # type: ignore[assignment]
        default_factory=list,
        max_length=5,
        description=(
            "当前素材中 0 到 5 个不同主体，按 salience 数值从高到低排列；"
            "每项同时给出 subject 和 description"
        ),
    )


class AssetUsageFeatureValue(FeatureValue):
    """Compatibility name for the now-uniform Feature envelope."""


def _legacy_feature_value(
    *,
    value: object,
    status: object = None,
    evidence: object = None,
) -> dict[str, object]:
    if status in {
        FeatureApplicability.UNKNOWN.value,
        FeatureApplicability.NOT_APPLICABLE.value,
    }:
        return {"applicability": status, "items": []}

    if isinstance(value, str):
        descriptions = [
            item.strip()[:120]
            for item in value.replace(";", "；").split("；")
            if item.strip()
        ]
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            return {
                "applicability": FeatureApplicability.APPLICABLE.value,
                "items": value[:5],
            }
        descriptions = [
            item.strip()[:120] for item in value if isinstance(item, str) and item.strip()
        ]
    else:
        descriptions = []

    descriptions = list(dict.fromkeys(descriptions))[:5]
    if not descriptions:
        return {"applicability": FeatureApplicability.UNKNOWN.value, "items": []}

    item_status = status if status in {item.value for item in FeatureStatus} else "inferred"
    if isinstance(evidence, str):
        normalized_evidence = [evidence.strip()[:80]] if evidence.strip() else []
    elif isinstance(evidence, list):
        normalized_evidence = [
            item.strip()[:80] for item in evidence if isinstance(item, str) and item.strip()
        ][:1]
    else:
        normalized_evidence = []
    salience_by_index = (
        FeatureSalience.HIGH.value,
        FeatureSalience.MEDIUM.value,
        FeatureSalience.MEDIUM.value,
        FeatureSalience.LOW.value,
        FeatureSalience.LOW.value,
    )
    return {
        "applicability": FeatureApplicability.APPLICABLE.value,
        "items": [
            {
                "description": description,
                "salience": salience_by_index[index],
                "status": item_status,
                "evidence": normalized_evidence if index == 0 else [],
                "ocr_confidence": None,
            }
            for index, description in enumerate(descriptions)
        ],
    }


class AssetFeatures(BaseModel):
    subject_content: SubjectFeatureValue = Field(
        description=FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT]
    )
    scene_theme: FeatureValue = Field(
        description=FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME]
    )
    visual_presentation: FeatureValue = Field(
        description=FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_PRESENTATION]
    )

    @model_validator(mode="before")
    @classmethod
    def fill_omitted_model_fields(cls, value: Any) -> Any:
        if isinstance(value, list):
            value = cls._object_from_feature_list(value)
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "visual_presentation" not in normalized:
            legacy_visual_items: list[Any] = []
            for legacy_name in ("visual_style", "color_composition"):
                legacy_value = normalized.get(legacy_name)
                if isinstance(legacy_value, dict):
                    converted = FeatureValue.model_validate(legacy_value)
                    legacy_visual_items.extend(
                        item.model_dump(mode="json") for item in converted.items[:3]
                    )
            if legacy_visual_items:
                normalized["visual_presentation"] = {
                    "applicability": FeatureApplicability.APPLICABLE.value,
                    "items": legacy_visual_items[:5],
                }
        for field_name in cls.model_fields:
            normalized.setdefault(
                field_name,
                {
                    "applicability": FeatureApplicability.UNKNOWN.value,
                    "items": [],
                },
            )
        return normalized

    @classmethod
    def _object_from_feature_list(cls, value: list[Any]) -> dict[str, Any]:
        identifier_fields = ("key", "name", "embedding_type")
        valid_feature_names = set(cls.model_fields)
        normalized: dict[str, Any] = {}
        if not value:
            raise ValueError(
                "features must be an object or a non-empty unambiguous keyed feature array"
            )
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(
                    "features must be an object or an unambiguous keyed feature array; "
                    f"item {index} is not an object"
                )
            identifiers: list[str] = []
            for identifier_field in identifier_fields:
                if identifier_field not in item:
                    continue
                identifier = item[identifier_field]
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError(
                        "features must be an object or an unambiguous keyed feature array; "
                        f"item {index} has an invalid {identifier_field}"
                    )
                identifiers.append(identifier.strip())
            if not identifiers:
                raise ValueError(
                    "features must be an object or an unambiguous keyed feature array; "
                    f"item {index} has no key, name, or embedding_type"
                )
            feature_names = set(identifiers)
            if len(feature_names) != 1:
                raise ValueError(
                    "features must be an object or an unambiguous keyed feature array; "
                    f"item {index} has conflicting identifiers"
                )
            feature_name = feature_names.pop()
            if feature_name not in valid_feature_names:
                raise ValueError(
                    "features must be an object or an unambiguous keyed feature array; "
                    f"item {index} identifies unknown feature {feature_name!r}"
                )
            if feature_name in normalized:
                raise ValueError(
                    "features must be an object or an unambiguous keyed feature array; "
                    f"feature {feature_name!r} appears more than once"
                )
            normalized[feature_name] = {
                key: item_value
                for key, item_value in item.items()
                if key not in identifier_fields
            }
        return normalized


class AssetUnderstanding(BaseModel):
    asset_name: str = Field(description="不超过 20 字的素材名称")
    asset_description: str = Field(description="40 到 120 字的素材整体客观描述")
    features: AssetFeatures = Field(description="三个独立语义维度的结构化描述")
    transcript: str | None = Field(
        default=None,
        description="仅音频素材填写的逐字转写；其他素材返回 null",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_generated_text(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        asset_name = normalized.get("asset_name")
        if isinstance(asset_name, str):
            normalized["asset_name"] = asset_name.strip()[:40]
        asset_description = normalized.get("asset_description")
        if isinstance(asset_description, str):
            normalized["asset_description"] = asset_description.strip()[:500]
        transcript = normalized.get("transcript")
        if isinstance(transcript, str):
            normalized["transcript"] = transcript.strip()[:100_000]
        return normalized


class EmbeddingResult(BaseModel):
    vector: list[float]
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ProcessingJobRecord(BaseModel):
    job_id: str
    workspace_id: str
    input_path: str
    total_count: int
    completed_count: int
    failed_count: int
    status: str
    current_stage: str
    error_info: list[dict[str, Any]] = Field(default_factory=list)
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)
    started_at: datetime | None
    completed_at: datetime | None


class ProcessingJobListResponse(BaseModel):
    items: list[ProcessingJobRecord] = Field(default_factory=list)
