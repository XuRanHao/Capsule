from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from capsule.enums import (
    AssetType,
    ClusterInternalVariance,
    ClusterRepresentativeRole,
    FeatureStatus,
    ProcessingStatus,
)


class SourceContext(BaseModel):
    text: str
    relation_type: str
    text_block_index: int | None = None


class DiscoveredFile(BaseModel):
    path: str
    relative_path: str
    extension: str
    size_bytes: int


class AssetDraft(BaseModel):
    asset_type: AssetType
    file_name: str
    source_locator: dict[str, Any] = Field(default_factory=dict)
    source_contexts: list[SourceContext] = Field(default_factory=list)
    raw_content: str | None = None
    file_info: dict[str, Any] = Field(default_factory=dict)
    derived_file_uri: str | None = None
    preview_uri: str | None = None


class AssetCreate(BaseModel):
    asset_id: str
    workspace_id: str
    source_file_id: str
    asset_type: AssetType
    file_name: str
    file_type: str
    asset_key: str
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
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    feature_revision: int = 1
    embedding_revision: int = 1


class AssetRecord(BaseModel):
    asset_id: str
    workspace_id: str
    source_file_id: str
    asset_type: AssetType
    file_name: str
    file_type: str
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


class StoredFileResult(BaseModel):
    source_file_id: str
    asset_ids: list[str] = Field(default_factory=list)


class FeatureValue(BaseModel):
    value: str | None
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
        evidence = normalized.get("evidence")
        if isinstance(evidence, str):
            normalized["evidence"] = [evidence]
        elif evidence is None:
            normalized["evidence"] = []
        valid_statuses = {status.value for status in FeatureStatus}
        if normalized.get("status") not in valid_statuses:
            normalized["status"] = (
                FeatureStatus.INFERRED.value
                if normalized.get("value")
                else FeatureStatus.UNKNOWN.value
            )
        return normalized


class AssetFeatures(BaseModel):
    subject_content: FeatureValue
    scene_theme: FeatureValue
    visual_style: FeatureValue
    color_composition: FeatureValue
    mood_atmosphere: FeatureValue
    character_state_or_psychology: FeatureValue
    asset_usage: FeatureValue
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
    started_at: datetime | None
    completed_at: datetime | None


class ProcessingJobListResponse(BaseModel):
    items: list[ProcessingJobRecord] = Field(default_factory=list)
