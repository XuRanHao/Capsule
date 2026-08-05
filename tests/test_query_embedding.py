import asyncio
import math
from collections.abc import Sequence

from capsule.config import Settings
from capsule.enums import EmbeddingType
from capsule.schemas import EmbeddingResult
from capsule.search.models import (
    QueryEnhancement,
    QueryType,
    SearchFilters,
    SearchRequest,
    VectorSearchHit,
)
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall


class FakeEmbeddingClient:
    def __init__(self, *, fail_joint: bool = False) -> None:
        self.fail_joint = fail_joint
        self.calls: list[str] = []

    async def embed_text(self, text: str) -> EmbeddingResult:
        self.calls.append(f"text:{text}")
        return EmbeddingResult(vector=[3.0, 4.0, 0.0], model="fake")

    async def embed_image(self, image_url: str) -> EmbeddingResult:
        self.calls.append(f"image:{image_url}")
        return EmbeddingResult(vector=[0.0, 2.0, 0.0], model="fake")

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult:
        self.calls.append(f"image_text:{image_url}:{text}")
        if self.fail_joint:
            raise NotImplementedError("joint embeddings unavailable")
        return EmbeddingResult(vector=[1.0, 1.0, 0.0], model="fake")


class ConcurrentEmbeddingClient(FakeEmbeddingClient):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_observed = 0

    async def embed_text(self, text: str) -> EmbeddingResult:
        return await self._run("text")

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult:
        return await self._run("image_text")

    async def _run(self, name: str) -> EmbeddingResult:
        self.calls.append(name)
        self.active += 1
        self.max_observed = max(self.max_observed, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return EmbeddingResult(vector=[1.0, 1.0, 0.0], model="fake")


def settings() -> Settings:
    return Settings(
        embedding_dimension=3,
        search_embedding_concurrency=2,
    )


async def test_text_query_defaults_to_native_content_only() -> None:
    client = FakeEmbeddingClient()
    service = QueryEmbeddingService(client, settings())

    plan = await service.embed(
        SearchRequest(
            workspace_id="workspace_demo",
            query_type=QueryType.TEXT,
            query_text="蓝紫色黄昏",
        )
    )

    assert [item.embedding_type for item in plan.vectors] == [
        EmbeddingType.NATIVE_MULTIMODAL,
    ]
    assert all(math.isclose(sum(value**2 for value in item.vector), 1.0) for item in plan.vectors)
    assert client.calls == ["text:蓝紫色黄昏"]
    assert plan.degraded is False


async def test_image_text_uses_joint_vector_and_semantic_text_channels() -> None:
    client = FakeEmbeddingClient()
    service = QueryEmbeddingService(client, settings())

    plan = await service.embed(
        SearchRequest(
            workspace_id="workspace_demo",
            query_type=QueryType.IMAGE_TEXT,
            query_text="更像黄昏",
            query_image_url="https://example.com/query.png",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.ASSET_DESCRIPTION,
                EmbeddingType.SUBJECT_CONTENT,
                EmbeddingType.VISUAL_STYLE,
                EmbeddingType.MOOD_ATMOSPHERE,
                EmbeddingType.TARGET_AUDIENCE,
            ],
        )
    )

    assert [item.channel for item in plan.vectors] == [
        "native_multimodal",
        "asset_description",
        "subject_content",
        "visual_style",
        "mood_atmosphere",
        "target_audience",
    ]
    assert plan.degraded is False
    assert sorted(client.calls) == [
        "image_text:https://example.com/query.png:更像黄昏",
        "text:更像黄昏",
    ]


async def test_image_text_falls_back_to_separate_vectors() -> None:
    client = FakeEmbeddingClient(fail_joint=True)
    service = QueryEmbeddingService(client, settings())

    plan = await service.embed(
        SearchRequest(
            workspace_id="workspace_demo",
            query_type=QueryType.IMAGE_TEXT,
            query_text="保留构图",
            query_image_url="https://example.com/query.png",
        )
    )

    assert plan.vectors[0].channel == "native_multimodal"
    assert plan.vectors[0].weight == 1.0
    assert plan.degraded is True
    assert "image:https://example.com/query.png" in client.calls


async def test_image_text_embeddings_run_concurrently() -> None:
    client = ConcurrentEmbeddingClient()
    service = QueryEmbeddingService(client, settings())

    await service.embed(
        SearchRequest(
            workspace_id="workspace_demo",
            query_type=QueryType.IMAGE_TEXT,
            query_text="并发查询",
            query_image_url="https://example.com/query.png",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.ASSET_DESCRIPTION,
            ],
        )
    )

    assert client.max_observed == 2


class DistinctQueryEnhancementClient:
    async def enhance_search_query(
        self,
        *,
        query_text: str,
        embedding_types: Sequence[EmbeddingType],
    ) -> QueryEnhancement:
        return QueryEnhancement(
            queries={
                EmbeddingType.VISUAL_STYLE: "动画视觉风格",
                EmbeddingType.COLOR_COMPOSITION: "蓝紫色调与黄昏光影",
            },
            weights={
                EmbeddingType.VISUAL_STYLE: 0.6,
                EmbeddingType.COLOR_COMPOSITION: 0.4,
            },
        )


class ProjectionEmbeddingClient(FakeEmbeddingClient):
    async def embed_text(self, text: str) -> EmbeddingResult:
        self.calls.append(f"text:{text}")
        vectors = {
            "动画视觉风格": [1.0, 0.0, 0.0],
            "蓝紫色调与黄昏光影": [0.0, 1.0, 0.0],
        }
        return EmbeddingResult(vector=vectors[text], model="fake")


class ProjectionRecallRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[float]]] = []

    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        filters: SearchFilters,
        limit: int,
    ) -> Sequence[VectorSearchHit]:
        self.calls.append((embedding_type, vector))
        return []


async def test_enhanced_queries_flow_to_independent_embedding_and_recall_routes() -> None:
    request = SearchRequest(
        workspace_id="workspace_demo",
        query_type=QueryType.TEXT,
        query_text="蓝紫色黄昏动画场景",
        embedding_types=[
            EmbeddingType.VISUAL_STYLE,
            EmbeddingType.COLOR_COMPOSITION,
        ],
    )
    parsed, reasons = await QueryParser(DistinctQueryEnhancementClient()).parse(
        request,
        image_url=None,
    )
    client = ProjectionEmbeddingClient()

    plan = await QueryEmbeddingService(client, settings()).embed(request, parsed)
    recall_repository = ProjectionRecallRepository()
    await MultiChannelRecall(recall_repository, settings()).search(
        plan=plan,
        workspace_id=request.workspace_id,
        filters=request.filters,
        top_k=request.top_k,
    )

    assert reasons == ()
    assert client.calls == [
        "text:动画视觉风格",
        "text:蓝紫色调与黄昏光影",
    ]
    assert [vector.weight for vector in plan.vectors] == [0.6, 0.4]
    assert recall_repository.calls == [
        ("visual_style", [1.0, 0.0, 0.0]),
        ("color_composition", [0.0, 1.0, 0.0]),
    ]


async def test_image_query_reuses_image_vector_for_selected_semantic_channels() -> None:
    client = FakeEmbeddingClient()
    service = QueryEmbeddingService(client, settings())

    plan = await service.embed(
        SearchRequest(
            workspace_id="workspace_demo",
            query_type=QueryType.IMAGE,
            query_image_url="https://example.com/query.png",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.VISUAL_STYLE,
            ],
        )
    )

    assert [item.embedding_type for item in plan.vectors] == [
        EmbeddingType.NATIVE_MULTIMODAL,
        EmbeddingType.VISUAL_STYLE,
    ]
    assert client.calls == ["image:https://example.com/query.png"]
    assert [item.weight for item in plan.vectors] == [0.5, 0.5]
