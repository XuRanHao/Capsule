from typing import Any

from pydantic import BaseModel, Field

from capsule.enums import AssetType, FeatureStatus


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
    raw_content: dict[str, Any] | None = None
    file_info: dict[str, Any] = Field(default_factory=dict)
    derived_file_uri: str | None = None
    preview_uri: str | None = None


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
