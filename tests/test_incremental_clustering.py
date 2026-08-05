from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from capsule.config import Settings
from capsule.db.repositories import ClusterBootstrapState
from capsule.enums import ClusterMemberSource, ClusterMode, EmbeddingType
from capsule.pipeline.incremental_clustering import (
    IncrementalAssignmentThresholds,
    IncrementalClusterCoordinator,
    IncrementalClusterService,
    cosine_similarity,
    evaluate_cluster_bootstrap,
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
    def __init__(
        self,
        *,
        bootstrap_state: ClusterBootstrapState,
    ) -> None:
        super().__init__(clusters=[], embeddings=[])
        self.bootstrap_state = bootstrap_state
        self.bootstrap_calls: list[dict[str, object]] = []

    async def get_cluster_bootstrap_state(
        self,
        **kwargs: object,
    ) -> ClusterBootstrapState:
        self.bootstrap_calls.append(dict(kwargs))
        return self.bootstrap_state


class FakeClusterRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, EmbeddingType, str]] = []

    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        trigger: str = "user",
        **_: object,
    ) -> object:
        self.calls.append((workspace_id, embedding_type, trigger))
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


def _bootstrap_state(
    *,
    has_baseline: bool = False,
    run_in_progress: bool = False,
    eligible_asset_count: int = 50,
) -> ClusterBootstrapState:
    return ClusterBootstrapState(
        has_baseline=has_baseline,
        run_in_progress=run_in_progress,
        eligible_asset_count=eligible_asset_count,
        latest_run_id=None,
        latest_sample_count=0,
    )


@pytest.mark.parametrize(
    ("state", "should_bootstrap"),
    [
        (_bootstrap_state(eligible_asset_count=49), False),
        (_bootstrap_state(eligible_asset_count=50), True),
        (_bootstrap_state(has_baseline=True, eligible_asset_count=500), False),
        (_bootstrap_state(run_in_progress=True, eligible_asset_count=500), False),
    ],
    ids=["below-minimum", "first-baseline", "baseline-exists", "run-in-progress"],
)
def test_bootstrap_requires_minimum_without_baseline_or_active_run(
    state: ClusterBootstrapState,
    should_bootstrap: bool,
) -> None:
    decision = evaluate_cluster_bootstrap(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        state=state,
        minimum_asset_count=50,
    )

    assert decision.should_bootstrap is should_bootstrap
    assert decision.eligible_asset_count == state.eligible_asset_count
    assert decision.embedding_type is EmbeddingType.VISUAL_STYLE


def test_cosine_similarity_rejects_invalid_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0], [1.0, 0.0]) is None
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) is None
    assert cosine_similarity([float("nan"), 1.0], [1.0, 0.0]) is None


@pytest.mark.asyncio
async def test_coordinator_schedules_first_bootstrap_independently_per_dimension() -> None:
    repository = FakeCoordinatorRepository(
        bootstrap_state=_bootstrap_state(eligible_asset_count=2)
    )
    runner = FakeClusterRunner()
    coordinator = IncrementalClusterCoordinator(
        settings=Settings(
            cluster_bootstrap_minimum_count=2,
            cluster_bootstrap_concurrency=1,
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

    assert visual.bootstrap_scheduled
    assert subject.bootstrap_scheduled
    assert runner.calls == [
        ("workspace_a", EmbeddingType.VISUAL_STYLE, "automatic_bootstrap"),
        ("workspace_a", EmbeddingType.SUBJECT_CONTENT, "automatic_bootstrap"),
    ]


@pytest.mark.asyncio
async def test_coordinator_never_reclusters_automatically_after_baseline() -> None:
    repository = FakeCoordinatorRepository(
        bootstrap_state=_bootstrap_state(
            has_baseline=True,
            eligible_asset_count=10_000,
        )
    )
    runner = FakeClusterRunner()
    coordinator = IncrementalClusterCoordinator(
        settings=Settings(cluster_bootstrap_minimum_count=2),
        assignment_service=IncrementalClusterService(
            repository=repository,
            vector_store=FakeVectorStore({}),
        ),
        repository=repository,
        cluster_runner=runner,
    )

    result = await coordinator.process_assets(
        workspace_id="workspace_a",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        asset_ids=["new_asset"],
    )
    await coordinator.close()

    assert not result.bootstrap.should_bootstrap
    assert not result.bootstrap_scheduled
    assert runner.calls == []
