from collections.abc import Mapping, Sequence

from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.schemas import EmbeddingResult
from capsule.search.models import (
    SearchAssetRecord,
    SearchRequest,
    VectorSearchHit,
)
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.recall import MultiChannelRecall
from capsule.search.service import SearchService


class FakeEmbeddingClient:
    async def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[1.0, 0.0, 0.0], model="fake")

    async def embed_image(self, image_url: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[0.0, 1.0, 0.0], model="fake")

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=[1.0, 1.0, 0.0], model="fake")


class FakeVectorRepository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        asset_types: tuple[str, ...],
        limit: int,
    ) -> Sequence[VectorSearchHit]:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "embedding_type": embedding_type,
                "asset_types": asset_types,
                "limit": limit,
            }
        )
        if embedding_type == "visual_style":
            raise RuntimeError("simulated channel outage")
        return [
            VectorSearchHit(
                embedding_id=f"emb_{index}_{embedding_type}",
                asset_id=f"asset_{index}",
                source_file_id="source_shared" if index <= 4 else "source_other",
                asset_type="image",
                embedding_revision=1,
                similarity=1.0 - index / 100,
            )
            for index in range(1, 6)
        ]


class FakeAssetRepository:
    def __init__(self) -> None:
        self.calls = 0

    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
    ) -> Mapping[str, SearchAssetRecord]:
        self.calls += 1
        return {
            asset_id: asset_record(
                asset_id,
                workspace_id,
                "source_shared" if int(asset_id.removeprefix("asset_")) <= 4 else "source_other",
            )
            for asset_id in asset_ids
        }


def asset_record(asset_id: str, workspace_id: str, source_file_id: str) -> SearchAssetRecord:
    return SearchAssetRecord(
        asset_id=asset_id,
        workspace_id=workspace_id,
        source_file_id=source_file_id,
        asset_type="image",
        asset_name=f"Asset {asset_id}",
        asset_description="黄昏动画场景",
        asset_features={"visual_style": {"value": "动画"}},
        source_contexts=[{"text": "午后-黄昏", "relation_type": "preceding_text"}],
        source_locator={"block_index": 3},
        preview_uri=f"s3://previews/{asset_id}.jpg",
        processing_status="completed",
        source_file_name="moodboard.md",
        source_file_type="markdown",
        source_relative_path="references/moodboard.md",
    )


def build_service() -> tuple[SearchService, FakeVectorRepository, FakeAssetRepository]:
    settings = Settings(
        embedding_dimension=3,
        search_channel_top_k_multiplier=3,
        search_channel_top_k_cap=100,
        search_candidate_cap=300,
        search_same_source_limit=3,
    )
    vectors = FakeVectorRepository()
    assets = FakeAssetRepository()
    service = SearchService(
        query_embedding=QueryEmbeddingService(FakeEmbeddingClient(), settings),
        recall=MultiChannelRecall(vectors, settings),
        assets=assets,
        settings=settings,
    )
    return service, vectors, assets


async def test_search_degrades_one_channel_and_caps_same_source() -> None:
    service, vectors, assets = build_service()

    response = await service.search(
        SearchRequest.model_validate(
            {
                "workspace_id": "workspace_demo",
                "query_type": "text",
                "query_text": "蓝紫色黄昏动画场景",
                "filters": {"asset_type": ["image"]},
                "top_k": 4,
            }
        )
    )

    assert response.degraded is True
    assert response.total == 4
    assert [item.asset_id for item in response.results] == [
        "asset_1",
        "asset_2",
        "asset_3",
        "asset_5",
    ]
    assert all(item.source_contexts for item in response.results)
    assert assets.calls == 1
    assert len(vectors.calls) == 4
    assert all(call["limit"] == 12 for call in vectors.calls)
    assert all(call["workspace_id"] == "workspace_demo" for call in vectors.calls)


async def test_search_api_returns_the_service_response() -> None:
    service, _, _ = build_service()
    app = create_app(search_service=service)
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search",
                json={
                    "workspace_id": "workspace_demo",
                    "query_type": "image",
                    "query_image_url": "https://example.com/query.png",
                    "filters": {"asset_type": ["image"]},
                    "top_k": 2,
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"]["query_type"] == "image"
    assert payload["total"] == 2
    assert payload["results"][0]["source_contexts"][0]["text"] == "午后-黄昏"


async def test_search_api_validates_query_inputs() -> None:
    service, _, _ = build_service()
    app = create_app(search_service=service)
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/search",
                json={
                    "workspace_id": "workspace_demo",
                    "query_type": "image_text",
                    "query_text": "缺少图片",
                },
            )

    assert response.status_code == 422


async def test_search_api_allows_configured_frontend_origin() -> None:
    service, _, _ = build_service()
    app = create_app(search_service=service)
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.options(
                "/api/v1/search",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
