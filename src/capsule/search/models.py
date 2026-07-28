from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from capsule.enums import AssetType, EmbeddingType


class QueryType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    IMAGE_TEXT = "image_text"


class SearchFilters(BaseModel):
    asset_type: list[AssetType] = Field(default_factory=list)


class SearchRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=64)
    query_type: QueryType
    query_text: str | None = Field(default=None, max_length=100_000)
    query_image_url: str | None = Field(default=None, max_length=16_384)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query_inputs(self) -> "SearchRequest":
        text = self.query_text.strip() if self.query_text else None
        image_url = self.query_image_url.strip() if self.query_image_url else None
        if self.query_type is QueryType.TEXT and not text:
            raise ValueError("query_text is required for a text query")
        if self.query_type is QueryType.IMAGE and not image_url:
            raise ValueError("query_image_url is required for an image query")
        if self.query_type is QueryType.IMAGE_TEXT and (not text or not image_url):
            raise ValueError("query_text and query_image_url are required for an image_text query")
        self.query_text = text
        self.query_image_url = image_url
        return self


class SearchQueryEcho(BaseModel):
    query_type: QueryType
    query_text: str | None
    query_image_url: str | None


class MatchedChannel(BaseModel):
    channel: str
    embedding_type: EmbeddingType
    rank: int
    similarity: float
    rrf_contribution: float


class SourceFileResult(BaseModel):
    source_file_id: str
    original_file_name: str
    file_type: str
    relative_path: str


class SearchResult(BaseModel):
    asset_id: str
    asset_type: AssetType
    asset_name: str | None
    asset_description: str | None
    asset_features: dict[str, Any] = Field(default_factory=dict)
    source_contexts: list[dict[str, Any]] = Field(default_factory=list)
    source_locator: dict[str, Any] = Field(default_factory=dict)
    preview_uri: str | None
    source_file: SourceFileResult
    score: float
    matched_channels: list[MatchedChannel]
    matched_feature: str | None = None


class SearchResponse(BaseModel):
    query: SearchQueryEcho
    total: int
    degraded: bool = False
    degraded_reasons: list[str] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)


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
    embedding_type: EmbeddingType
    rank: int
    similarity: float
    rrf_contribution: float


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
    source_file_id: str
    asset_type: str
    asset_name: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    source_contexts: list[dict[str, Any]]
    source_locator: dict[str, Any]
    preview_uri: str | None
    processing_status: str
    source_file_name: str
    source_file_type: str
    source_relative_path: str
