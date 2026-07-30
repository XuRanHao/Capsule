import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import SecretStr

from capsule.config import Settings
from capsule.db.repositories import ClusterEmbeddingAsset, ClusterMembershipWrite
from capsule.enums import ClusterInternalVariance, ClusterRunStatus, EmbeddingType
from capsule.pipeline import cluster_service as cluster_service_module
from capsule.pipeline.cluster_service import ClusterService
from capsule.pipeline.clustering import ClusterResult, HdbscanParameters
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
        pca_dimension=2,
        min_cluster_size=4,
        min_samples=2,
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
        runs_by_type["native_multimodal"]["preprocessing"]["parameter_selection"] == "user_defined"
    )
    assert runs_by_type["native_multimodal"]["parameters"]["candidates_evaluated"] == 1
    assert (
        runs_by_type["visual_style"]["preprocessing"]["parameter_selection"]
        == "user_defined_selection_optimized"
    )
    assert runs_by_type["visual_style"]["parameters"]["candidates_evaluated"] > 1
    assert runs_by_type["visual_style"]["preprocessing"]["pca_dimension"] == 2
    assert runs_by_type["visual_style"]["parameters"]["min_cluster_size"] == 4
    assert runs_by_type["visual_style"]["parameters"]["min_samples"] == 2
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


@pytest.mark.asyncio
async def test_asset_usage_capsules_persist_member_path_context() -> None:
    embedding_type = EmbeddingType.ASSET_USAGE
    assets = _assets(embedding_type)
    repository = FakeClusterRepository()
    summary_client = FakeSummaryClient()
    service = ClusterService(
        settings=Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_model="test-embedding",
            embedding_dimension=2,
            milvus_collection="cluster-test",
        ),
        embedding_repository=FakeEmbeddingRepository({embedding_type: assets}),  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        vector_store=FakeVectorStore(
            {asset.embedding_id: _vector(index) for index, asset in enumerate(assets)}
        ),
        model_client=summary_client,
    )

    result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=embedding_type,
    )

    assert result.status == ClusterRunStatus.COMPLETED
    assert repository.capsules
    assert all("海报/素材/" in capsule.summary.description for capsule in repository.capsules)
    for messages in summary_client.prompts:
        payload = json.loads(str(messages[1]["content"]))
        assert payload["member_source_context"]["directory_counts"]
        assert payload["member_source_context"]["representative_files"]


@pytest.mark.asyncio
async def test_cluster_service_clusters_fewer_than_fifteen_vectors() -> None:
    embedding_type = EmbeddingType.NATIVE_MULTIMODAL
    assets = _assets(embedding_type, count=12)
    vectors = {
        asset.embedding_id: (
            [1.0 + index * 0.001, 0.1 + (index % 2) * 0.001]
            if index < 6
            else [0.1 + (index % 2) * 0.001, 1.0 + (index - 6) * 0.001]
        )
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
        embedding_repository=FakeEmbeddingRepository({embedding_type: assets}),  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        vector_store=FakeVectorStore(vectors),
        model_client=summary_client,
    )

    result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=embedding_type,
    )

    assert result.status == ClusterRunStatus.COMPLETED
    assert result.vector_count == 12
    assert result.cluster_count > 0
    run = next(iter(repository.runs.values()))
    assert run["parameters"]["min_cluster_size"] == 3
    assert run["parameters"]["min_samples"] == 1
    assert len(next(iter(repository.memberships.values()))) == 12


@pytest.mark.asyncio
async def test_cluster_service_merges_capsules_but_preserves_raw_hdbscan_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_type = EmbeddingType.TARGET_AUDIENCE
    assets = _assets(embedding_type, count=9)
    angles = [-2, 0, 2, 18, 20, 22, 55, 57, 59]
    matrix = np.asarray(
        [[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))] for angle in angles],
        dtype=np.float32,
    )
    raw_labels = np.repeat(np.asarray([0, 1, 2], dtype=np.int_), 3)

    def fake_cluster_vectors(*_: object, **__: object) -> ClusterResult:
        return ClusterResult(
            labels=raw_labels,
            probabilities=np.ones(9, dtype=np.float64),
            transformed_vectors=matrix,
            pca_dimension=2,
            parameters=HdbscanParameters(min_cluster_size=3, min_samples=1),
            quality_score=0.8,
            parameter_candidates_evaluated=1,
        )

    monkeypatch.setattr(cluster_service_module, "cluster_vectors", fake_cluster_vectors)
    repository = FakeClusterRepository()
    service = ClusterService(
        settings=Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_model="test-embedding",
            embedding_dimension=2,
            milvus_collection="cluster-test",
        ),
        embedding_repository=FakeEmbeddingRepository({embedding_type: assets}),  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        vector_store=FakeVectorStore(
            {asset.embedding_id: matrix[index].tolist() for index, asset in enumerate(assets)}
        ),
        model_client=FakeSummaryClient(),
    )

    result = await service.run(
        workspace_id="workspace_cluster_service",
        embedding_type=embedding_type,
    )

    assert result.status == ClusterRunStatus.COMPLETED
    assert result.cluster_count == 2
    assert sorted(capsule.member_count for capsule in repository.capsules) == [3, 6]

    run = next(iter(repository.runs.values()))
    merge_metadata = run["parameters"]["semantic_merge"]
    assert merge_metadata["raw_cluster_count"] == 3
    assert merge_metadata["merged_cluster_count"] == 2
    assert merge_metadata["raw_to_merged_labels"] == {"0": 0, "1": 0, "2": 2}

    memberships = next(iter(repository.memberships.values()))
    assert {membership.hdbscan_label for membership in memberships} == {0, 1, 2}
    capsule_ids_by_raw_label = {
        label: {
            membership.cluster_capsule_id
            for membership in memberships
            if membership.hdbscan_label == label
        }
        for label in {0, 1, 2}
    }
    assert capsule_ids_by_raw_label[0] == capsule_ids_by_raw_label[1]
    assert capsule_ids_by_raw_label[0] != capsule_ids_by_raw_label[2]


def _assets(
    embedding_type: EmbeddingType,
    *,
    count: int = 20,
) -> list[ClusterEmbeddingAsset]:
    return [
        ClusterEmbeddingAsset(
            embedding_id=f"emb_{embedding_type.value}_{index}",
            asset_id=f"asset_{embedding_type.value}_{index}",
            source_file_id=f"src_{index}",
            asset_type="image",
            asset_name=f"测试资产 {index}",
            asset_description=f"测试资产 {index} 的描述。",
            asset_features={
                "visual_style": {"value": "测试风格"},
                "asset_usage": {
                    "value": "海报制作",
                    "status": "metadata",
                    "description": (
                        f"该素材对应相对文件路径「海报/素材/{index}.png」，用于海报制作。"
                    ),
                    "source_path": f"海报/素材/{index}.png",
                },
            },
            file_tree_context=["test"],
            source_relative_path=f"海报/素材/{index}.png",
        )
        for index in range(count)
    ]


def _vector(index: int) -> list[float]:
    if index < 10:
        return [1.0 + index * 0.001, 0.1 + (index % 2) * 0.001]
    return [0.1 + (index % 2) * 0.001, 1.0 + (index - 10) * 0.001]
