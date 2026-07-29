from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

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


class StoredFileResult(BaseModel):
    source_file_id: str
    asset_ids: list[str] = Field(default_factory=list)


class FeatureValue(BaseModel):
    value: str | None
    status: FeatureStatus
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


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
