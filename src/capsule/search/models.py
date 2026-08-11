import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from capsule.enums import AssetType, EmbeddingType
from capsule.features import embedding_type_supports_any_asset_type


class QueryType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"


class FusionMethod(StrEnum):
    WEIGHTED_RRF = "weighted_rrf"
    NORMALIZED_WEIGHTED_SIMILARITY = "normalized_weighted_similarity"


class QueryDimensionSource(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    JOINT = "joint"


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_id: str | None = Field(default=None, max_length=64)
    asset_type: list[AssetType] = Field(default_factory=list)
    file_type: list[str] = Field(default_factory=list)
    source_file_id: list[str] = Field(default_factory=list)
    created_at_from: datetime | None = None
    created_at_to: datetime | None = None
    model_name: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("model_name", "embedding_model_version"),
    )
    favorite: bool | None = None
    cluster_capsule_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_created_at_range(self) -> "SearchFilters":
        if (
            self.created_at_from is not None
            and self.created_at_to is not None
            and self.created_at_from > self.created_at_to
        ):
            raise ValueError("created_at_from must not be after created_at_to")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=64)
    created_by: str = Field(default="user_demo", min_length=1, max_length=128)
    query_type: QueryType
    query_text: str | None = Field(default=None, max_length=100_000)
    query_image_url: str | None = Field(default=None, max_length=16_384)
    query_image_upload_id: str | None = Field(default=None, max_length=128)
    embedding_types: list[EmbeddingType] = Field(
        default_factory=lambda: [EmbeddingType.NATIVE_MULTIMODAL],
        min_length=1,
        max_length=len(EmbeddingType),
    )
    dimension_weights: dict[EmbeddingType, float] | None = None
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_RRF
    save_capsule: bool = False
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query_inputs(self) -> "SearchRequest":
        text = self.query_text.strip() if self.query_text else None
        image_url = self.query_image_url.strip() if self.query_image_url else None
        upload_id = self.query_image_upload_id.strip() if self.query_image_upload_id else None
        has_image = bool(image_url or upload_id)
        if self.query_type is QueryType.TEXT and not text:
            raise ValueError("query_text is required for a text query")
        if self.query_type is QueryType.IMAGE and not has_image:
            raise ValueError(
                "query_image_url or query_image_upload_id is required for an image query"
            )
        if self.query_type is QueryType.IMAGE_TEXT and (not text or not has_image):
            raise ValueError(
                "query_text and one of query_image_url/query_image_upload_id "
                "are required for an image_text query"
            )
        if image_url and upload_id:
            raise ValueError("provide only one of query_image_url and query_image_upload_id")
        self.query_text = text
        self.query_image_url = image_url
        self.query_image_upload_id = upload_id
        if len(self.embedding_types) != len(set(self.embedding_types)):
            raise ValueError("embedding_types must contain unique values")
        if self.dimension_weights is not None:
            if set(self.dimension_weights) != set(self.embedding_types):
                raise ValueError(
                    "dimension_weights keys must exactly match embedding_types"
                )
            if any(
                not math.isfinite(weight) or weight <= 0
                for weight in self.dimension_weights.values()
            ):
                raise ValueError("dimension_weights must be positive finite numbers")
            total_weight = sum(self.dimension_weights.values())
            self.dimension_weights = {
                embedding_type: weight / total_weight
                for embedding_type, weight in self.dimension_weights.items()
            }
        unsupported = [
            embedding_type.value
            for embedding_type in self.embedding_types
            if not embedding_type_supports_any_asset_type(
                embedding_type=embedding_type,
                asset_types=self.filters.asset_type,
            )
        ]
        if unsupported:
            targets = ", ".join(item.value for item in self.filters.asset_type)
            raise ValueError(
                f"embedding_types {', '.join(unsupported)} are not supported "
                f"for target asset types {targets}"
            )
        return self

class DimensionQuery(BaseModel):
    embedding_type: EmbeddingType
    query: str = Field(min_length=1, max_length=8_000)
    weight: float = Field(gt=0, le=1)
    source: QueryDimensionSource = QueryDimensionSource.TEXT


class QueryEnhancement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: dict[EmbeddingType, str] = Field(min_length=1, max_length=12)
    weights: dict[EmbeddingType, float] = Field(min_length=1, max_length=12)


class SearchDimensionSuggestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=1, max_length=100_000)
    asset_types: list[AssetType] = Field(min_length=1, max_length=len(AssetType))

    @model_validator(mode="after")
    def normalize_input(self) -> "SearchDimensionSuggestionRequest":
        self.query_text = self.query_text.strip()
        if not self.query_text:
            raise ValueError("query_text must not be blank")
        if len(self.asset_types) != len(set(self.asset_types)):
            raise ValueError("asset_types must contain unique values")
        return self


class SearchDimensionSuggestionResponse(BaseModel):
    embedding_types: list[EmbeddingType] = Field(min_length=1, max_length=4)
    weights: dict[EmbeddingType, float] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_unique_dimensions(self) -> "SearchDimensionSuggestionResponse":
        if len(self.embedding_types) != len(set(self.embedding_types)):
            raise ValueError("embedding_types must contain unique values")
        if set(self.weights) != set(self.embedding_types):
            raise ValueError("weights keys must exactly match embedding_types")
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in self.weights.values()
        ):
            raise ValueError("weights must be positive finite numbers")
        total_weight = sum(self.weights.values())
        self.weights = {
            embedding_type: weight / total_weight
            for embedding_type, weight in self.weights.items()
        }
        return self


class ParsedQuery(BaseModel):
    dimension_queries: list[DimensionQuery] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ParsedQuery":
        types = [item.embedding_type for item in self.dimension_queries]
        if len(types) != len(set(types)):
            raise ValueError("dimension_queries must contain unique embedding types")
        weight_sum = sum(item.weight for item in self.dimension_queries)
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError("dimension query weights must sum to 1")
        return self


class SearchQueryEcho(BaseModel):
    query_type: QueryType
    query_text: str | None
    query_image_url: str | None
    query_image_upload_id: str | None = None
    embedding_types: list[EmbeddingType] = Field(
        default_factory=lambda: [EmbeddingType.NATIVE_MULTIMODAL]
    )


class MatchedChannel(BaseModel):
    channel: str
    embedding_type: EmbeddingType | None = None
    embedding_id: str | None = None
    embedding_revision: int | None = None
    rank: int
    similarity: float
    fusion_contribution: float = 0.0
    rrf_contribution: float = 0.0


class SourceFileResult(BaseModel):
    source_file_id: str
    original_file_name: str
    file_type: str
    relative_path: str
    workspace_id: str | None = None
    project_id: str | None = None
    mime_type: str | None = None
    file_tree_context: list[str] = Field(default_factory=list)
    storage_uri: str | None = None
    sha256: str | None = None
    file_size_bytes: int | None = None
    processing_status: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SearchResult(BaseModel):
    asset_id: str
    asset_type: AssetType
    workspace_id: str | None = None
    project_id: str | None = None
    source_file_id: str | None = None
    file_name: str | None = None
    file_type: str | None = None
    asset_key: str | None = None
    content_hash: str | None = None
    asset_name: str | None
    asset_name_source: str | None = None
    asset_description: str | None
    asset_features: dict[str, Any] = Field(default_factory=dict)
    file_tree_context: list[str] = Field(default_factory=list)
    source_contexts: list[dict[str, Any]] = Field(default_factory=list)
    file_info: dict[str, Any] = Field(default_factory=dict)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: str | None = None
    derived_file_uri: str | None = None
    preview_uri: str | None = None
    processing_status: str | None = None
    feature_revision: int | None = None
    embedding_revision: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    source_file: SourceFileResult | None = None
    score: float
    matched_channels: list[MatchedChannel]
    matched_feature: str | None = None
    matched_reason: str | None = None
    group_kind: str | None = None
    folded_asset_ids: list[str] = Field(default_factory=list)
    available: bool = True
    # Hierarchical-index metadata lets callers identify the parent document and
    # preserve a stable order among its retrievable children.
    parent_asset_id: str | None = None
    index_role: str = "standalone"
    child_order: int | None = None


class SearchTimings(BaseModel):
    query_enhancement_ms: float = 0
    embedding_ms: float = 0
    recall_ms: float = 0
    fusion_ms: float = 0
    hydration_ms: float = 0
    cluster_ms: float = 0
    total_ms: float = 0


class ClusterSearchResult(BaseModel):
    cluster_capsule_id: str
    cluster_run_id: str
    embedding_type: EmbeddingType
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    common_features: list[str] = Field(default_factory=list)
    member_count: int
    average_membership_probability: float
    medoid_asset_id: str | None = None
    representative_asset_ids: list[str] = Field(default_factory=list)
    matched_asset_ids: list[str] = Field(default_factory=list)
    matched_asset_count: int = 0
    score: float


class SearchResponse(BaseModel):
    query: SearchQueryEcho
    parsed_query: ParsedQuery | None = None
    fusion_method: FusionMethod = FusionMethod.WEIGHTED_RRF
    search_engine_version: str = "search-v1"
    execution_id: str | None = None
    capsule_id: str | None = None
    total: int
    asset_total: int = 0
    cluster_total: int = 0
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    timings: SearchTimings = Field(default_factory=SearchTimings)
    assets: list[SearchResult] = Field(default_factory=list)
    clusters: list[ClusterSearchResult] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)


class QueryImageUploadResponse(BaseModel):
    upload_id: str
    image_url: str


class CapsuleSnapshot(BaseModel):
    execution_id: str
    created_at: datetime
    results: list[SearchResult] = Field(default_factory=list)


class SearchCapsuleSummary(BaseModel):
    capsule_id: str
    workspace_id: str
    created_by: str
    query_type: QueryType
    query_text: str | None
    query_image_uri: str | None
    fusion_method: FusionMethod
    is_favorite: bool
    result_count: int
    last_used_at: datetime
    created_at: datetime


class SearchCapsuleListResponse(BaseModel):
    items: list[SearchCapsuleSummary] = Field(default_factory=list)


class SearchCapsuleDetail(SearchCapsuleSummary):
    parsed_query: ParsedQuery
    filters: SearchFilters
    search_engine_version: str
    embedding_model: str
    latest_snapshot: CapsuleSnapshot
    executions: list[str] = Field(default_factory=list)


class SearchCapsulePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_favorite: bool


@dataclass(slots=True, frozen=True)
class QueryVector:
    channel: str
    embedding_type: EmbeddingType
    vector: list[float]
    weight: float


@dataclass(slots=True, frozen=True)
class QueryEmbeddingPlan:
    vectors: tuple[QueryVector, ...]
    degraded: bool = False
    degraded_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class VectorSearchHit:
    embedding_id: str
    asset_id: str
    source_file_id: str
    asset_type: str
    embedding_revision: int
    similarity: float


@dataclass(slots=True, frozen=True)
class TextSearchHit:
    asset_id: str
    source_file_id: str
    asset_type: str
    score: float


@dataclass(slots=True, frozen=True)
class ChannelRecall:
    query_vector: QueryVector
    hits: tuple[VectorSearchHit, ...]


@dataclass(slots=True, frozen=True)
class RecallBatch:
    channels: tuple[ChannelRecall, ...]
    degraded: bool = False
    degraded_reasons: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class ChannelMatch:
    channel: str
    embedding_type: EmbeddingType | None
    embedding_id: str | None
    embedding_revision: int | None
    rank: int
    similarity: float
    fusion_contribution: float = 0.0
    rrf_contribution: float = 0.0


@dataclass(slots=True)
class FusedHit:
    asset_id: str
    source_file_id: str
    asset_type: str
    score: float = 0.0
    matched_channels: list[ChannelMatch] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class SearchAssetRecord:
    asset_id: str
    workspace_id: str
    project_id: str
    source_file_id: str
    asset_type: str
    file_name: str
    file_type: str
    asset_key: str
    content_hash: str
    asset_name: str | None
    asset_name_source: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    file_tree_context: list[str]
    source_contexts: list[dict[str, Any]]
    file_info: dict[str, Any]
    source_locator: dict[str, Any]
    raw_content: str | None
    derived_file_uri: str | None
    preview_uri: str | None
    processing_status: str
    feature_revision: int
    embedding_revision: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    source_workspace_id: str
    source_project_id: str
    source_file_name: str
    source_file_type: str
    source_mime_type: str
    source_relative_path: str
    source_file_tree_context: list[str]
    source_storage_uri: str
    source_sha256: str
    source_file_size_bytes: int
    source_processing_status: str
    source_error_message: str | None
    source_created_at: datetime
    source_updated_at: datetime
    indexed_embedding_ids: frozenset[str]
    # Keep these trailing defaults so existing repository fakes and hand-built
    # records remain source-compatible while hierarchy indexing rolls out.
    parent_asset_id: str | None = None
    index_role: str = "standalone"
    child_order: int | None = None
