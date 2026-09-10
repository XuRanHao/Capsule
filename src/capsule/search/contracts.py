from collections.abc import Mapping, Sequence
from typing import Protocol

from capsule.enums import AssetType, EmbeddingType
from capsule.schemas import EmbeddingResult
from capsule.search.models import (
    QueryEnhancement,
    SearchAssetRecord,
    SearchDimensionSuggestionResponse,
    SearchFilters,
    TextSearchHit,
    VectorSearchHit,
)


class QueryEmbeddingClient(Protocol):
    async def embed_text(self, text: str) -> EmbeddingResult: ...

    async def embed_image(self, image_url: str) -> EmbeddingResult: ...

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult: ...


class SearchUnderstandingClient(Protocol):
    async def select_search_dimensions(
        self,
        *,
        query_text: str,
        asset_types: Sequence[AssetType],
    ) -> SearchDimensionSuggestionResponse: ...

    async def enhance_search_query(
        self,
        *,
        query_text: str,
        embedding_types: Sequence[EmbeddingType],
    ) -> QueryEnhancement: ...


class QueryImageResolver(Protocol):
    async def resolve(
        self,
        *,
        workspace_id: str,
        upload_id: str,
    ) -> str: ...


class VectorSearchRepository(Protocol):
    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        filters: SearchFilters,
        limit: int,
    ) -> Sequence[VectorSearchHit]: ...


class SearchVectorIndexPreparer(Protocol):
    async def ensure_search_vectors(
        self,
        *,
        workspace_id: str,
        embedding_types: Sequence[EmbeddingType],
    ) -> Mapping[EmbeddingType, str]: ...


class AssetSearchRepository(Protocol):
    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
        embedding_ids: Sequence[str],
        created_by: str = "user_demo",
        filters: SearchFilters | None = None,
    ) -> Mapping[str, SearchAssetRecord]: ...

    async def get_children_by_parent_ids(
        self,
        *,
        workspace_id: str,
        parent_asset_ids: Sequence[str],
    ) -> Mapping[str, SearchAssetRecord]: ...


class TextSearchRepository(Protocol):
    async def search_text(
        self,
        *,
        workspace_id: str,
        query_text: str,
        filters: SearchFilters,
        created_by: str,
        limit: int,
    ) -> Sequence[TextSearchHit]: ...
