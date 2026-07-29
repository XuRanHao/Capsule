from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from capsule.enums import AssetType, FeatureStatus, ProcessingStatus


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
    name: str
    description: str
    keywords: list[str] = Field(min_length=1, max_length=8)
    common_features: list[str] = Field(default_factory=list)
    internal_variance: str
