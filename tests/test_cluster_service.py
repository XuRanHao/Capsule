import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from capsule.config import Settings
from capsule.db.repositories import ClusterEmbeddingAsset, ClusterMembershipWrite
from capsule.enums import ClusterInternalVariance, ClusterRunStatus, EmbeddingType
from capsule.pipeline.cluster_service import ClusterService
from capsule.schemas import ClusterCapsuleWrite, ClusterSummary


class FakeEmbeddingRepository:
    def __init__(self, assets_by_type: Mapping[EmbeddingType, list[ClusterEmbeddingAsset]]) -> None:
        self.assets_by_type = assets_by_type

    async def list_indexed_cluster_embeddings(
        self,
        *,
        embedding_type: str,
        **_: object,
    ) -> list[ClusterEmbeddingAsset]:
        return list(self.assets_by_type.get(EmbeddingType(embedding_type), []))


class FakeClusterRepository:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.memberships: dict[str, list[ClusterMembershipWrite]] = {}
        self.capsules: list[ClusterCapsuleWrite] = []

    async def create_run(self, **values: Any) -> str:
        run_id = f"run_{len(self.runs)}"
        self.runs[run_id] = values
        return run_id

    async def complete_run(self, *, cluster_run_id: str, **values: Any) -> None:
        self.runs[cluster_run_id].update(values)

    async def fail_run(self, *, cluster_run_id: str, error: str) -> None:
        self.runs[cluster_run_id]["error"] = error

    async def upsert_capsule(self, values: ClusterCapsuleWrite) -> SimpleNamespace:
        self.capsules.append(values)
        return SimpleNamespace(cluster_capsule_id=f"capsule_{len(self.capsules)}")

    async def store_memberships(
        self,
        *,
        cluster_run_id: str,
        memberships: list[ClusterMembershipWrite],
    ) -> None:
        self.memberships[cluster_run_id] = memberships


class FakeVectorStore:
    def __init__(self, vectors: Mapping[str, list[float]]) -> None:
        self.vectors = vectors
        self.ensure_calls = 0

    async def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return False

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]:
        return {
            embedding_id: self.vectors[embedding_id]
            for embedding_id in embedding_ids
            if embedding_id in self.vectors
        }


class FakeSummaryClient:
    def __init__(self) -> None:
        self.prompts: list[list[Mapping[str, Any]]] = []

    async def summarize_cluster(self, messages: Sequence[Mapping[str, Any]]) -> ClusterSummary:
        self.prompts.append(list(messages))
        return ClusterSummary(
            name="测试聚类",
            description=(
                "这一组测试资产在向量空间中具有稳定的共同特征，代表资产显示出相近的内容和"
                "视觉表现，少量边缘样本不影响该组的整体判断。"
            ),
            keywords=["测试", "聚类", "代表资产"],
            common_features=["向量相近"],
            internal_variance=ClusterInternalVariance.LOW,
        )


@pytest.mark.asyncio
async def test_cluster_service_runs_each_requested_embedding_type_independently() -> None:
    embedding_types = [EmbeddingType.NATIVE_MULTIMODAL, EmbeddingType.VISUAL_STYLE]
    assets_by_type = {embedding_type: _assets(embedding_type) for embedding_type in embedding_types}
    vectors = {
        asset.embedding_id: _vector(index)
        for assets in assets_by_type.values()
        for index, asset in enumerate(assets)
    }
    repository = FakeClusterRepository()
    summary_client = FakeSummaryClient()
    service = ClusterService(
        settings=Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_model="test-embedding",
            embedding_dimension=2,
            milvus_collection="cluster-test",
        ),
        embedding_repository=FakeEmbeddingRepository(assets_by_type),  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        vector_store=FakeVectorStore(vectors),
        model_client=summary_client,
    )

    native_result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
    )
    visual_result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        optimize_parameters=True,
    )

    assert native_result.embedding_type == EmbeddingType.NATIVE_MULTIMODAL
    assert visual_result.embedding_type == EmbeddingType.VISUAL_STYLE
    assert native_result.status == ClusterRunStatus.COMPLETED
    assert visual_result.status == ClusterRunStatus.COMPLETED
    assert len(repository.runs) == 2
    assert {run["embedding_type"] for run in repository.runs.values()} == {
        "native_multimodal",
        "visual_style",
    }
    runs_by_type = {run["embedding_type"]: run for run in repository.runs.values()}
    assert (
        runs_by_type["native_multimodal"]["preprocessing"]["parameter_selection"]
        == "size_based_default"
    )
    assert runs_by_type["native_multimodal"]["parameters"]["candidates_evaluated"] == 1
    assert (
        runs_by_type["visual_style"]["preprocessing"]["parameter_selection"]
        == "adaptive_dbcv_silhouette"
    )
    assert runs_by_type["visual_style"]["parameters"]["candidates_evaluated"] > 1
    assert all(len(memberships) == 20 for memberships in repository.memberships.values())
    assert {capsule.embedding_type for capsule in repository.capsules} == {
        "native_multimodal",
        "visual_style",
    }
    assert summary_client.prompts
    for messages in summary_client.prompts:
        payload = json.loads(str(messages[1]["content"]))
        assert 1 <= len(payload["representative_assets"]) <= 10


@pytest.mark.asyncio
async def test_cluster_service_records_insufficient_type_without_model_call() -> None:
    repository = FakeClusterRepository()
    summary_client = FakeSummaryClient()
    service = ClusterService(
        settings=Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_model="test-embedding",
            embedding_dimension=2,
            milvus_collection="cluster-test",
        ),
        embedding_repository=FakeEmbeddingRepository({}),  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        vector_store=FakeVectorStore({}),
        model_client=summary_client,
    )

    result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=EmbeddingType.ASSET_USAGE,
    )

    assert result.status == ClusterRunStatus.INSUFFICIENT_DATA
    assert result.vector_count == 0
    assert not summary_client.prompts
    assert next(iter(repository.runs.values()))["status"] == ClusterRunStatus.INSUFFICIENT_DATA


def _assets(embedding_type: EmbeddingType) -> list[ClusterEmbeddingAsset]:
    return [
        ClusterEmbeddingAsset(
            embedding_id=f"emb_{embedding_type.value}_{index}",
            asset_id=f"asset_{embedding_type.value}_{index}",
            source_file_id=f"src_{index}",
            asset_type="image",
            asset_name=f"测试资产 {index}",
            asset_description=f"测试资产 {index} 的描述。",
            asset_features={"visual_style": {"value": "测试风格"}},
            file_tree_context=["test"],
        )
        for index in range(20)
    ]


def _vector(index: int) -> list[float]:
    if index < 10:
        return [1.0 + index * 0.001, 0.1 + (index % 2) * 0.001]
    return [0.1 + (index % 2) * 0.001, 1.0 + (index - 10) * 0.001]
