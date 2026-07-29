from collections.abc import Mapping, Sequence
from typing import Protocol

from capsule.schemas import EmbeddingResult
from capsule.search.models import (
    ParsedQuery,
    RerankBatch,
    SearchAssetRecord,
    SearchFilters,
    SearchRequest,
    VectorSearchHit,
)


class QueryEmbeddingClient(Protocol):
    async def embed_text(self, text: str) -> EmbeddingResult: ...

    async def embed_image(self, image_url: str) -> EmbeddingResult: ...

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult: ...


class SearchUnderstandingClient(Protocol):
    async def parse_search_query(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
    ) -> ParsedQuery: ...

    async def rerank_search_results(
        self,
        request: SearchRequest,
        *,
        image_url: str | None,
        candidates: Sequence[Mapping[str, object]],
    ) -> RerankBatch: ...


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
