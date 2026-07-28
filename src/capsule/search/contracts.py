from collections.abc import Mapping, Sequence
from typing import Protocol

from capsule.schemas import EmbeddingResult
from capsule.search.models import SearchAssetRecord, VectorSearchHit


class QueryEmbeddingClient(Protocol):
    async def embed_text(self, text: str) -> EmbeddingResult: ...

    async def embed_image(self, image_url: str) -> EmbeddingResult: ...

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult: ...


class VectorSearchRepository(Protocol):
    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        asset_types: tuple[str, ...],
        limit: int,
    ) -> Sequence[VectorSearchHit]: ...


class AssetSearchRepository(Protocol):
    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
    ) -> Mapping[str, SearchAssetRecord]: ...
