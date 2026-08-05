from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capsule.enums import (
    AssetIndexRole,
    AssetType,
    ClusterInternalVariance,
    ClusterMemberSource,
    ClusterMode,
    ClusterRepresentativeRole,
    EmbeddingType,
    FeatureStatus,
    NewAssetClusterStatus,
    ProcessingStatus,
)


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
    source_file: AssetSourceRecord
    embeddings: list[AssetEmbeddingState] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AssetListResponse(BaseModel):
    items: list[AssetViewRecord] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


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


class FeatureValue(BaseModel):
    value: str | None = Field(
        description=(
            "该维度内最多五个互不重复的名词、形容词或短语，使用中文分号连接；无法确定时为 null"
        )
    )
    status: FeatureStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_model_output(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {
                "value": value,
                "status": FeatureStatus.INFERRED.value,
                "confidence": 0.5,
                "evidence": [],
            }
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        raw_value = normalized.get("value")
        if isinstance(raw_value, list):
            terms = [
                item.strip()[:32] for item in raw_value if isinstance(item, str) and item.strip()
            ]
            normalized["value"] = "；".join(dict.fromkeys(terms[:5])) or None
        elif isinstance(raw_value, str):
            terms = [
                item.strip()[:32]
                for item in raw_value.replace(";", "；").split("；")
                if item.strip()
            ]
            normalized["value"] = "；".join(dict.fromkeys(terms[:5])) or None
        evidence = normalized.get("evidence")
        if isinstance(evidence, str):
            normalized["evidence"] = [evidence.strip()[:80]] if evidence.strip() else []
        elif isinstance(evidence, list):
            normalized["evidence"] = [
                item.strip()[:80] for item in evidence if isinstance(item, str) and item.strip()
            ][:1]
        elif evidence is None:
            normalized["evidence"] = []
        valid_statuses = {status.value for status in FeatureStatus}
        if normalized.get("status") not in valid_statuses:
            normalized["status"] = (
                FeatureStatus.INFERRED.value
                if normalized.get("value")
                else FeatureStatus.UNKNOWN.value
            )
        if normalized.get("status") in {
            FeatureStatus.UNKNOWN.value,
            FeatureStatus.NOT_APPLICABLE.value,
        }:
            normalized["value"] = None
        return normalized


class AssetUsageFeatureValue(FeatureValue):
    """Usage semantics plus the deterministic relative-path evidence behind them."""

    description: str | None = Field(
        default=None,
        max_length=500,
        description="明确包含相对文件路径的素材用途说明",
    )
    source_path: str | None = Field(
        default=None,
        max_length=2048,
        description="用于判断素材用途的 Workspace 相对路径",
    )


class AssetFeatures(BaseModel):
    subject_content: FeatureValue
    scene_theme: FeatureValue
    visual_style: FeatureValue
    color_composition: FeatureValue
    mood_atmosphere: FeatureValue
    character_state_or_psychology: FeatureValue
    asset_usage: AssetUsageFeatureValue
    target_audience: FeatureValue
    provenance: FeatureValue
    rights_version_authorship: FeatureValue

    @model_validator(mode="before")
    @classmethod
    def fill_omitted_model_fields(cls, value: Any) -> Any:
        if isinstance(value, list):
            raise ValueError("features must be an object, not a list")
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in cls.model_fields:
            normalized.setdefault(
                field_name,
                {
                    "value": None,
                    "status": FeatureStatus.UNKNOWN.value,
                    "confidence": 0.0,
                    "evidence": [],
                },
            )
        return normalized


class AssetUnderstanding(BaseModel):
    asset_name: str
    asset_description: str
    features: AssetFeatures

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
        return normalized


class EmbeddingResult(BaseModel):
    vector: list[float]
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class ClusterSummary(BaseModel):
    name: str = Field(min_length=1, max_length=1024)
    description: str = Field(min_length=50, max_length=150)
    keywords: list[str] = Field(min_length=3, max_length=8)
    common_features: list[str] = Field(default_factory=list, max_length=8)
    internal_variance: ClusterInternalVariance


class ClusterRepresentativeWrite(BaseModel):
    """One persisted representative Asset. `asset_id` is always an Asset primary key."""

    asset_id: str = Field(min_length=1, max_length=64)
    role: ClusterRepresentativeRole
    rank: int = Field(ge=0)
    distance_to_medoid: float = Field(ge=0)
    membership_probability: float = Field(ge=0, le=1)


class ClusterCapsuleWrite(BaseModel):
    """Model-generated Cluster Capsule fields and selected representative Asset IDs."""

    cluster_run_id: str = Field(min_length=1, max_length=64)
    workspace_id: str = Field(min_length=1, max_length=64)
    embedding_type: str = Field(min_length=1, max_length=64)
    cluster_label: int = Field(ge=0)
    summary: ClusterSummary
    member_count: int = Field(ge=1)
    average_membership_probability: float = Field(ge=0, le=1)
    representatives: list[ClusterRepresentativeWrite] = Field(min_length=1, max_length=10)


class ClusterCapsuleRecord(BaseModel):
    cluster_capsule_id: str
    cluster_run_id: str
    workspace_id: str
    embedding_type: str
    cluster_label: int
    model_generated_name: str
    user_override_name: str | None
    effective_name: str
    model_generated_description: str
    user_override_description: str | None
    effective_description: str
    keywords: list[str] = Field(default_factory=list)
    common_features: list[str] = Field(default_factory=list)
    internal_variance: ClusterInternalVariance | None
    member_count: int
    average_membership_probability: float
    medoid_asset_id: str | None
    representative_asset_ids: list[str] = Field(default_factory=list)
    is_favorite: bool


class ClusterRunRecord(BaseModel):
    cluster_run_id: str
    workspace_id: str
    embedding_type: str
    input_embedding_ids: list[str] = Field(default_factory=list)
    dataset_hash: str
    sample_count: int
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    cluster_count: int | None
    noise_count: int | None
    noise_ratio: float | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None


class ClusterMemberRecord(BaseModel):
    asset_id: str
    asset_type: AssetType
    file_name: str
    asset_name: str | None
    asset_description: str | None
    source_file_id: str
    relative_path: str
    hdbscan_label: int
    membership_probability: float
    is_noise: bool
    distance_to_representative: float | None
    preview_url: str | None = None


class ClusterRunListResponse(BaseModel):
    items: list[ClusterRunRecord] = Field(default_factory=list)


class CurrentClusterRecord(BaseModel):
    """One currently active logical cluster, independent of a historical run snapshot."""

    model_config = ConfigDict(from_attributes=True)

    cluster_id: str
    workspace_id: str
    embedding_type: str
    mode: ClusterMode
    name: str
    description: str
    representative_asset_id: str | None
    source_run_id: str | None
    created_at: datetime
    updated_at: datetime


class CurrentClusterMemberRecord(BaseModel):
    """One Asset's current assignment for an embedding dimension."""

    model_config = ConfigDict(from_attributes=True)

    cluster_id: str
    asset_id: str
    embedding_type: str
    source: ClusterMemberSource
    score: float | None
    created_at: datetime


class CurrentClusterListResponse(BaseModel):
    items: list[CurrentClusterRecord] = Field(default_factory=list)


class CurrentClusterMemberListResponse(BaseModel):
    items: list[CurrentClusterMemberRecord] = Field(default_factory=list)


class NewAssetClusterStatusItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    asset_type: AssetType
    file_name: str
    asset_name: str | None = None
    status: NewAssetClusterStatus
    cluster_id: str | None = None
    cluster_name: str | None = None
    cluster_mode: ClusterMode | None = None
    member_source: ClusterMemberSource | None = None
    score: float | None = None
    created_at: datetime


class NewAssetClusterStatusResponse(BaseModel):
    workspace_id: str
    embedding_type: EmbeddingType
    initialized: bool
    bootstrap_minimum_count: int = Field(ge=1)
    baseline_cluster_run_id: str | None = None
    baseline_sample_count: int | None = Field(default=None, ge=0)
    eligible_asset_count: int = Field(ge=0)
    new_asset_count: int = Field(ge=0)
    incrementally_clustered_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    manual_management_count: int = Field(ge=0)
    items: list[NewAssetClusterStatusItem] = Field(default_factory=list)


class CurrentClusterPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ClusterMode


AssetId = Annotated[str, Field(min_length=1, max_length=64)]


class CurrentClusterMemberMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_ids: list[AssetId] = Field(min_length=1, max_length=500)

    @field_validator("asset_ids")
    @classmethod
    def deduplicate_asset_ids(cls, asset_ids: list[str]) -> list[str]:
        return list(dict.fromkeys(asset_ids))


class CurrentClusterMemberMutationResponse(BaseModel):
    cluster_id: str
    asset_ids: list[str] = Field(default_factory=list)


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
