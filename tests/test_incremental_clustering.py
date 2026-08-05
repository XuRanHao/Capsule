from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from capsule.config import Settings
from capsule.db.repositories import CurrentClusterReclusterCounts
from capsule.enums import ClusterMemberSource, ClusterMode, EmbeddingType
from capsule.pipeline.incremental_clustering import (
    IncrementalAssignmentThresholds,
    IncrementalClusterCoordinator,
    IncrementalClusterService,
    ReclusterThreshold,
    cosine_similarity,
    evaluate_recluster_threshold,
)


@dataclass(frozen=True)
class FakeCluster:
    cluster_id: str
    mode: ClusterMode
    representative_asset_id: str | None


@dataclass(frozen=True)
class FakeEmbedding:
    asset_id: str
    embedding_id: str


class FakeRepository:
    def __init__(
        self,
        *,
        clusters: Sequence[FakeCluster],
        embeddings: Sequence[FakeEmbedding],
        excluded_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        self.clusters = list(clusters)
        self.embeddings = list(embeddings)
        self.excluded_pairs = excluded_pairs or set()
        self.list_cluster_calls: list[dict[str, object]] = []
        self.embedding_calls: list[dict[str, object]] = []
        self.exclusion_calls: list[dict[str, object]] = []
        self.attach_calls: list[dict[str, object]] = []

    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: Sequence[ClusterMode] | None = None,
    ) -> list[FakeCluster]:
        self.list_cluster_calls.append(
            {
                "workspace_id": workspace_id,
                "embedding_type": embedding_type,
                "modes": tuple(modes or ()),
            }
        )
        selected_modes = set(modes or ())
        return [cluster for cluster in self.clusters if cluster.mode in selected_modes]

    async def list_indexed_asset_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        asset_ids: Sequence[str],
    ) -> list[FakeEmbedding]:
        self.embedding_calls.append(
            {
                "workspace_id": workspace_id,
                "embedding_type": embedding_type,
                "asset_ids": tuple(asset_ids),
            }
        )
        selected_ids = set(asset_ids)
        return [item for item in self.embeddings if item.asset_id in selected_ids]

    async def list_excluded_pairs(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        cluster_ids: Sequence[str],
        asset_ids: Sequence[str],
    ) -> set[tuple[str, str]]:
        self.exclusion_calls.append(
            {
                "workspace_id": workspace_id,
                "embedding_type": embedding_type,
                "cluster_ids": tuple(cluster_ids),
                "asset_ids": tuple(asset_ids),
            }
        )
        return self.excluded_pairs & {
            (cluster_id, asset_id) for cluster_id in cluster_ids for asset_id in asset_ids
        }

    async def attach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: Sequence[str],
        source: ClusterMemberSource,
        scores: dict[str, float] | None = None,
    ) -> list[object]:
        self.attach_calls.append(
            {
                "cluster_id": cluster_id,
                "workspace_id": workspace_id,
                "asset_ids": tuple(asset_ids),
                "source": source,
                "scores": dict(scores or {}),
            }
        )
        return []


class FakeCoordinatorRepository(FakeRepository):
    async def get_recluster_counts(self, **_: object) -> CurrentClusterReclusterCounts:
        return CurrentClusterReclusterCounts(
            baseline_dynamic_count=10,
            delta_dynamic_count=2,
        )


class FakeClusterRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, EmbeddingType]] = []

    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        **_: object,
    ) -> object:
        self.calls.append((workspace_id, embedding_type))
        return object()


class FakeVectorStore:
    def __init__(self, vectors: Mapping[str, list[float]]) -> None:
        self.vectors = dict(vectors)
        self.ensure_calls = 0
        self.fetch_calls: list[tuple[str, ...]] = []

    async def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return False

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]:
        self.fetch_calls.append(tuple(embedding_ids))
        return {
            embedding_id: self.vectors[embedding_id]
            for embedding_id in embedding_ids
            if embedding_id in self.vectors
        }


@pytest.mark.asyncio
async def test_resident_open_is_preferred_over_more_similar_dynamic_cluster() -> None:
    repository = FakeRepository(
        clusters=[
            FakeCluster("resident", ClusterMode.RESIDENT_OPEN, "resident_rep"),
            FakeCluster("dynamic", ClusterMode.DYNAMIC, "dynamic_rep"),
            FakeCluster("manual", ClusterMode.RESIDENT_MANUAL, "manual_rep"),
        ],
        embeddings=[
            FakeEmbedding("new_asset", "emb_new"),
            FakeEmbedding("resident_rep", "emb_resident"),
            FakeEmbedding("dynamic_rep", "emb_dynamic"),
            FakeEmbedding("manual_rep", "emb_manual"),
        ],
    )
    service = IncrementalClusterService(
        repository=repository,
        vector_store=FakeVectorStore(
            {
                "emb_new": [1.0, 0.0],
                "emb_resident": [0.95, 0.31],
                "emb_dynamic": [1.0, 0.0],
                "emb_manual": [1.0, 0.0],
            }
        ),
        default_thresholds=IncrementalAssignmentThresholds(
            resident_open=0.9,
            dynamic=0.9,
        ),
    )

    result = await service.assign_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        asset_ids=["new_asset"],
    )

    assert result.assigned_count == 1
    assert result.assignments[0].cluster_id == "resident"
    assert result.resident_assigned_count == 1
    assert repository.list_cluster_calls == [
        {
            "workspace_id": "workspace_a",
            "embedding_type": "visual_style",
            "modes": (ClusterMode.RESIDENT_OPEN, ClusterMode.DYNAMIC),
        }
    ]
    assert repository.attach_calls[0]["source"] is ClusterMemberSource.INCREMENTAL
    assert repository.attach_calls[0]["asset_ids"] == ("new_asset",)


@pytest.mark.asyncio
async def test_exclusion_blocks_resident_and_falls_back_to_dynamic() -> None:
    repository = FakeRepository(
        clusters=[
            FakeCluster("resident", ClusterMode.RESIDENT_OPEN, "resident_rep"),
            FakeCluster("dynamic", ClusterMode.DYNAMIC, "dynamic_rep"),
        ],
        embeddings=[
            FakeEmbedding("new_asset", "emb_new"),
            FakeEmbedding("resident_rep", "emb_resident"),
            FakeEmbedding("dynamic_rep", "emb_dynamic"),
        ],
        excluded_pairs={("resident", "new_asset")},
    )
    service = IncrementalClusterService(
        repository=repository,
        vector_store=FakeVectorStore(
            {
                "emb_new": [1.0, 0.0],
                "emb_resident": [1.0, 0.0],
                "emb_dynamic": [0.98, 0.2],
            }
        ),
        default_thresholds=IncrementalAssignmentThresholds(0.9, 0.9),
    )

    result = await service.assign_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.SUBJECT_CONTENT,
        asset_ids=["new_asset"],
    )

    assert result.assignments[0].cluster_id == "dynamic"
    assert result.dynamic_assigned_count == 1
    assert repository.attach_calls[0]["cluster_id"] == "dynamic"


@pytest.mark.asyncio
async def test_no_qualifying_candidate_leaves_asset_pending_without_write() -> None:
    repository = FakeRepository(
        clusters=[FakeCluster("dynamic", ClusterMode.DYNAMIC, "representative")],
        embeddings=[
            FakeEmbedding("new_asset", "emb_new"),
            FakeEmbedding("representative", "emb_rep"),
        ],
    )
    service = IncrementalClusterService(
        repository=repository,
        vector_store=FakeVectorStore(
            {"emb_new": [1.0, 0.0], "emb_rep": [0.0, 1.0]}
        ),
        default_thresholds=IncrementalAssignmentThresholds(0.95, 0.95),
    )

    result = await service.assign_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.SUBJECT_CONTENT,
        asset_ids=["new_asset"],
    )

    assert result.assignments == ()
    assert result.pending_asset_ids == ("new_asset",)
    assert result.missing_vector_asset_ids == ()
    assert repository.attach_calls == []


@pytest.mark.asyncio
async def test_missing_or_invalid_asset_vectors_are_reported_and_not_pending() -> None:
    repository = FakeRepository(
        clusters=[FakeCluster("dynamic", ClusterMode.DYNAMIC, "representative")],
        embeddings=[
            FakeEmbedding("invalid_asset", "emb_invalid"),
            FakeEmbedding("representative", "emb_rep"),
        ],
    )
    service = IncrementalClusterService(
        repository=repository,
        vector_store=FakeVectorStore(
            {"emb_invalid": [0.0, 0.0], "emb_rep": [1.0, 0.0]}
        ),
    )

    result = await service.assign_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.ASSET_USAGE,
        asset_ids=["missing_asset", "invalid_asset", "missing_asset"],
    )

    assert result.requested_asset_ids == ("missing_asset", "invalid_asset")
    assert result.missing_vector_asset_ids == ("missing_asset", "invalid_asset")
    assert result.pending_asset_ids == ()
    assert repository.attach_calls == []


@pytest.mark.asyncio
async def test_embedding_type_specific_threshold_is_used_for_one_dimension() -> None:
    repository = FakeRepository(
        clusters=[FakeCluster("dynamic", ClusterMode.DYNAMIC, "representative")],
        embeddings=[
            FakeEmbedding("new_asset", "emb_new"),
            FakeEmbedding("representative", "emb_rep"),
        ],
    )
    service = IncrementalClusterService(
        repository=repository,
        vector_store=FakeVectorStore(
            {"emb_new": [1.0, 0.0], "emb_rep": [0.9, 0.435889894]}
        ),
        default_thresholds=IncrementalAssignmentThresholds(0.95, 0.95),
        thresholds_by_embedding_type={
            EmbeddingType.MOOD_ATMOSPHERE: IncrementalAssignmentThresholds(0.85, 0.85)
        },
    )

    result = await service.assign_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.MOOD_ATMOSPHERE,
        asset_ids=["new_asset"],
    )

    assert result.assigned_count == 1
    assert repository.list_cluster_calls[0]["embedding_type"] == "mood_atmosphere"
    assert repository.embedding_calls[0]["embedding_type"] == "mood_atmosphere"
    assert repository.exclusion_calls[0]["embedding_type"] == "mood_atmosphere"


def test_recluster_condition_requires_ratio_and_minimum_for_one_dimension() -> None:
    threshold = ReclusterThreshold(ratio=0.1, minimum=50)

    below_minimum = evaluate_recluster_threshold(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        baseline_dynamic_count=100,
        delta_dynamic_count=20,
        threshold=threshold,
    )
    below_ratio = evaluate_recluster_threshold(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        baseline_dynamic_count=1000,
        delta_dynamic_count=50,
        threshold=threshold,
    )
    reached = evaluate_recluster_threshold(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        baseline_dynamic_count=500,
        delta_dynamic_count=50,
        threshold=threshold,
    )

    assert below_minimum.delta_ratio == pytest.approx(0.2)
    assert not below_minimum.should_recluster
    assert not below_ratio.should_recluster
    assert reached.should_recluster
    assert reached.embedding_type is EmbeddingType.VISUAL_STYLE


def test_recluster_condition_handles_empty_baseline_without_cross_type_state() -> None:
    decision = evaluate_recluster_threshold(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        baseline_dynamic_count=0,
        delta_dynamic_count=10,
        threshold=ReclusterThreshold(ratio=0.1, minimum=10),
    )

    assert decision.delta_ratio == 10.0
    assert decision.should_recluster
    assert decision.workspace_id == "workspace_a"


def test_cosine_similarity_rejects_invalid_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0], [1.0, 0.0]) is None
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None
    assert cosine_similarity([float("nan"), 1.0], [1.0, 0.0]) is None


@pytest.mark.asyncio
async def test_coordinator_schedules_recluster_independently_per_dimension() -> None:
    repository = FakeCoordinatorRepository(clusters=[], embeddings=[])
    runner = FakeClusterRunner()
    coordinator = IncrementalClusterCoordinator(
        settings=Settings(
            cluster_recluster_ratio_threshold=0.1,
            cluster_recluster_minimum_count=2,
            cluster_recluster_concurrency=1,
        ),
        assignment_service=IncrementalClusterService(
            repository=repository,
            vector_store=FakeVectorStore({}),
        ),
        repository=repository,
        cluster_runner=runner,
    )

    visual = await coordinator.process_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        asset_ids=["asset_visual"],
    )
    subject = await coordinator.process_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.SUBJECT_CONTENT,
        asset_ids=["asset_subject"],
    )
    await coordinator.close()

    assert visual.recluster_scheduled
    assert subject.recluster_scheduled
    assert runner.calls == [
        ("workspace_a", EmbeddingType.VISUAL_STYLE),
        ("workspace_a", EmbeddingType.SUBJECT_CONTENT),
    ]
