"""Assign newly embedded Assets to current clusters without rebuilding a dimension.

The service deliberately processes exactly one ``workspace_id`` and one
``EmbeddingType`` per call.  Persistence and vector retrieval are expressed as
small protocols so assignment policy can be tested without PostgreSQL or
Milvus.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Protocol

from capsule.config import Settings
from capsule.db.repositories import CurrentClusterReclusterCounts
from capsule.enums import ClusterMemberSource, ClusterMode, EmbeddingType

logger = logging.getLogger(__name__)


class IncrementalClusterCandidate(Protocol):
    @property
    def cluster_id(self) -> str: ...

    @property
    def mode(self) -> ClusterMode | str: ...

    @property
    def representative_asset_id(self) -> str | None: ...


class IncrementalEmbedding(Protocol):
    @property
    def asset_id(self) -> str: ...

    @property
    def embedding_id(self) -> str: ...


class IncrementalClusterRepository(Protocol):
    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: Sequence[ClusterMode] | None = None,
    ) -> Sequence[IncrementalClusterCandidate]: ...

    async def list_indexed_asset_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        asset_ids: Sequence[str],
    ) -> Sequence[IncrementalEmbedding]: ...

    async def list_excluded_pairs(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        cluster_ids: Sequence[str],
        asset_ids: Sequence[str],
    ) -> Set[tuple[str, str]]: ...

    async def attach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: Sequence[str],
        source: ClusterMemberSource,
        scores: dict[str, float] | None = None,
    ) -> Sequence[object]: ...


class IncrementalVectorStore(Protocol):
    async def ensure_collection(self) -> bool: ...

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]: ...


class ReclusterCountRepository(Protocol):
    async def get_recluster_counts(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> CurrentClusterReclusterCounts: ...


class FullClusterRunner(Protocol):
    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL,
        cluster_run_id: str | None = None,
        pca_dimension: int = 8,
        min_samples: int = 1,
        min_cluster_size: int = 3,
        optimize_parameters: bool = False,
    ) -> object: ...


@dataclass(slots=True, frozen=True)
class IncrementalAssignmentThresholds:
    """Cosine thresholds for the two algorithm-addressable cluster modes."""

    resident_open: float = 0.92
    dynamic: float = 0.92

    def __post_init__(self) -> None:
        for field_name, value in (
            ("resident_open", self.resident_open),
            ("dynamic", self.dynamic),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{field_name} cosine threshold must be between -1 and 1")

    def for_mode(self, mode: ClusterMode) -> float:
        if mode is ClusterMode.RESIDENT_OPEN:
            return self.resident_open
        if mode is ClusterMode.DYNAMIC:
            return self.dynamic
        raise ValueError(f"incremental assignment does not support cluster mode {mode.value}")


@dataclass(slots=True, frozen=True)
class IncrementalClusterAssignment:
    asset_id: str
    cluster_id: str
    cluster_mode: ClusterMode
    score: float


@dataclass(slots=True, frozen=True)
class IncrementalAssignmentResult:
    workspace_id: str
    embedding_type: EmbeddingType
    requested_asset_ids: tuple[str, ...]
    assignments: tuple[IncrementalClusterAssignment, ...]
    pending_asset_ids: tuple[str, ...]
    missing_vector_asset_ids: tuple[str, ...]

    @property
    def assigned_count(self) -> int:
        return len(self.assignments)

    @property
    def resident_assigned_count(self) -> int:
        return sum(
            assignment.cluster_mode is ClusterMode.RESIDENT_OPEN
            for assignment in self.assignments
        )

    @property
    def dynamic_assigned_count(self) -> int:
        return sum(
            assignment.cluster_mode is ClusterMode.DYNAMIC for assignment in self.assignments
        )


@dataclass(slots=True, frozen=True)
class ReclusterThreshold:
    """Trigger policy for one independent embedding dimension."""

    ratio: float
    minimum: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.ratio) or not 0.0 <= self.ratio <= 1.0:
            raise ValueError("recluster ratio threshold must be between 0 and 1")
        if self.minimum < 1:
            raise ValueError("recluster minimum must be at least 1")


@dataclass(slots=True, frozen=True)
class ReclusterDecision:
    workspace_id: str
    embedding_type: EmbeddingType
    baseline_dynamic_count: int
    delta_dynamic_count: int
    delta_ratio: float
    should_recluster: bool


@dataclass(slots=True, frozen=True)
class IncrementalClusterProcessResult:
    assignment: IncrementalAssignmentResult
    recluster: ReclusterDecision
    recluster_scheduled: bool


class IncrementalClusterService:
    """Assign unclustered Assets using representative-Asset cosine similarity."""

    def __init__(
        self,
        *,
        repository: IncrementalClusterRepository,
        vector_store: IncrementalVectorStore,
        default_thresholds: IncrementalAssignmentThresholds | None = None,
        thresholds_by_embedding_type: Mapping[
            EmbeddingType | str, IncrementalAssignmentThresholds
        ]
        | None = None,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._default_thresholds = default_thresholds or IncrementalAssignmentThresholds()
        self._thresholds_by_embedding_type = {
            _embedding_type_value(key): value
            for key, value in (thresholds_by_embedding_type or {}).items()
        }

    async def assign_assets(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        asset_ids: Sequence[str],
    ) -> IncrementalAssignmentResult:
        """Assign Assets in one dimension, preferring resident-open clusters.

        A qualifying resident-open candidate always wins over a dynamic
        candidate, even if the latter has a higher score.  Resident-manual
        clusters are never requested as candidates.  Repository writes remain
        authoritative: they atomically preserve an existing user membership.
        """

        requested_asset_ids = tuple(dict.fromkeys(asset_ids))
        if not requested_asset_ids:
            return IncrementalAssignmentResult(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                requested_asset_ids=(),
                assignments=(),
                pending_asset_ids=(),
                missing_vector_asset_ids=(),
            )

        embedding_type_value = embedding_type.value
        raw_candidates = await self._repository.list_clusters(
            workspace_id=workspace_id,
            embedding_type=embedding_type_value,
            modes=(ClusterMode.RESIDENT_OPEN, ClusterMode.DYNAMIC),
        )
        candidates = [
            candidate
            for candidate in raw_candidates
            if _candidate_mode(candidate) in {ClusterMode.RESIDENT_OPEN, ClusterMode.DYNAMIC}
            and candidate.representative_asset_id is not None
        ]
        representative_asset_ids = tuple(
            dict.fromkeys(
                candidate.representative_asset_id
                for candidate in candidates
                if candidate.representative_asset_id is not None
            )
        )
        embedding_assets = await self._repository.list_indexed_asset_embeddings(
            workspace_id=workspace_id,
            embedding_type=embedding_type_value,
            asset_ids=(*requested_asset_ids, *representative_asset_ids),
        )
        embedding_id_by_asset = {item.asset_id: item.embedding_id for item in embedding_assets}
        target_embedding_ids = {
            asset_id: embedding_id_by_asset[asset_id]
            for asset_id in requested_asset_ids
            if asset_id in embedding_id_by_asset
        }
        missing_embedding_assets = {
            asset_id for asset_id in requested_asset_ids if asset_id not in target_embedding_ids
        }

        requested_embedding_ids = tuple(dict.fromkeys(embedding_id_by_asset.values()))
        vectors: dict[str, list[float]] = {}
        if requested_embedding_ids:
            await self._vector_store.ensure_collection()
            vectors = await self._vector_store.fetch_vectors(requested_embedding_ids)

        target_vectors = {
            asset_id: vectors[embedding_id]
            for asset_id, embedding_id in target_embedding_ids.items()
            if embedding_id in vectors and _valid_vector(vectors[embedding_id])
        }
        missing_vector_assets = missing_embedding_assets | {
            asset_id for asset_id in target_embedding_ids if asset_id not in target_vectors
        }
        usable_candidates = [
            candidate
            for candidate in candidates
            if (
                candidate.representative_asset_id in embedding_id_by_asset
                and embedding_id_by_asset[candidate.representative_asset_id] in vectors
                and _valid_vector(
                    vectors[embedding_id_by_asset[candidate.representative_asset_id]]
                )
            )
        ]
        excluded_pairs = await self._repository.list_excluded_pairs(
            workspace_id=workspace_id,
            embedding_type=embedding_type_value,
            cluster_ids=tuple(candidate.cluster_id for candidate in usable_candidates),
            asset_ids=tuple(target_vectors),
        )
        thresholds = self._thresholds_by_embedding_type.get(
            embedding_type_value,
            self._default_thresholds,
        )

        planned: list[IncrementalClusterAssignment] = []
        for asset_id in requested_asset_ids:
            target_vector = target_vectors.get(asset_id)
            if target_vector is None:
                continue
            assignment = _select_assignment(
                asset_id=asset_id,
                vector=target_vector,
                candidates=usable_candidates,
                embedding_id_by_asset=embedding_id_by_asset,
                vectors=vectors,
                excluded_pairs=excluded_pairs,
                thresholds=thresholds,
            )
            if assignment is not None:
                planned.append(assignment)

        assignments_by_cluster: dict[str, list[IncrementalClusterAssignment]] = defaultdict(list)
        for assignment in planned:
            assignments_by_cluster[assignment.cluster_id].append(assignment)
        for cluster_id, assignments in assignments_by_cluster.items():
            await self._repository.attach_members(
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                asset_ids=tuple(assignment.asset_id for assignment in assignments),
                source=ClusterMemberSource.INCREMENTAL,
                scores={assignment.asset_id: assignment.score for assignment in assignments},
            )

        assigned_ids = {assignment.asset_id for assignment in planned}
        return IncrementalAssignmentResult(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            requested_asset_ids=requested_asset_ids,
            assignments=tuple(planned),
            pending_asset_ids=tuple(
                asset_id
                for asset_id in requested_asset_ids
                if asset_id not in assigned_ids and asset_id not in missing_vector_assets
            ),
            missing_vector_asset_ids=tuple(
                asset_id for asset_id in requested_asset_ids if asset_id in missing_vector_assets
            ),
        )


class IncrementalClusterCoordinator:
    """Assign one dimension, then independently schedule its full rebuild if due."""

    def __init__(
        self,
        *,
        settings: Settings,
        assignment_service: IncrementalClusterService,
        repository: ReclusterCountRepository,
        cluster_runner: FullClusterRunner,
    ) -> None:
        self._settings = settings
        self._assignment_service = assignment_service
        self._repository = repository
        self._cluster_runner = cluster_runner
        self._semaphore = asyncio.Semaphore(settings.cluster_recluster_concurrency)
        self._running_keys: set[tuple[str, EmbeddingType]] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def process_assets(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        asset_ids: list[str],
    ) -> IncrementalClusterProcessResult:
        assignment = await self._assignment_service.assign_assets(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            asset_ids=asset_ids,
        )
        counts = await self._repository.get_recluster_counts(
            workspace_id=workspace_id,
            embedding_type=embedding_type.value,
            model_name=self._settings.embedding_model,
            dimension=self._settings.embedding_dimension,
            milvus_collection=self._settings.milvus_collection,
        )
        decision = evaluate_recluster_threshold(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            baseline_dynamic_count=counts.baseline_dynamic_count,
            delta_dynamic_count=counts.delta_dynamic_count,
            threshold=ReclusterThreshold(
                ratio=self._settings.cluster_recluster_ratio_threshold,
                minimum=self._settings.cluster_recluster_minimum_count,
            ),
        )
        key = (workspace_id, embedding_type)
        scheduled = decision.should_recluster and key not in self._running_keys
        if scheduled:
            self._running_keys.add(key)
            task = asyncio.create_task(self._run_recluster(key))
            self._tasks.add(task)
            task.add_done_callback(self._recluster_done)
        return IncrementalClusterProcessResult(
            assignment=assignment,
            recluster=decision,
            recluster_scheduled=scheduled,
        )

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_recluster(self, key: tuple[str, EmbeddingType]) -> None:
        workspace_id, embedding_type = key
        try:
            async with self._semaphore:
                await self._cluster_runner.run(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type,
                )
        finally:
            self._running_keys.discard(key)

    def _recluster_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("background dimension reclustering failed")


def evaluate_recluster_threshold(
    *,
    workspace_id: str,
    embedding_type: EmbeddingType,
    baseline_dynamic_count: int,
    delta_dynamic_count: int,
    threshold: ReclusterThreshold,
) -> ReclusterDecision:
    """Evaluate one dimension only; callers must invoke it separately per type."""

    if baseline_dynamic_count < 0:
        raise ValueError("baseline_dynamic_count cannot be negative")
    if delta_dynamic_count < 0:
        raise ValueError("delta_dynamic_count cannot be negative")
    ratio = delta_dynamic_count / max(1, baseline_dynamic_count)
    return ReclusterDecision(
        workspace_id=workspace_id,
        embedding_type=embedding_type,
        baseline_dynamic_count=baseline_dynamic_count,
        delta_dynamic_count=delta_dynamic_count,
        delta_ratio=ratio,
        should_recluster=(
            delta_dynamic_count >= threshold.minimum and ratio >= threshold.ratio
        ),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return cosine similarity, or ``None`` for unusable/mismatched vectors."""

    if len(left) != len(right) or not _valid_vector(left) or not _valid_vector(right):
        return None
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    score = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


def _select_assignment(
    *,
    asset_id: str,
    vector: Sequence[float],
    candidates: Sequence[IncrementalClusterCandidate],
    embedding_id_by_asset: Mapping[str, str],
    vectors: Mapping[str, list[float]],
    excluded_pairs: Set[tuple[str, str]],
    thresholds: IncrementalAssignmentThresholds,
) -> IncrementalClusterAssignment | None:
    for mode in (ClusterMode.RESIDENT_OPEN, ClusterMode.DYNAMIC):
        best: IncrementalClusterAssignment | None = None
        for candidate in candidates:
            if _candidate_mode(candidate) is not mode:
                continue
            if (candidate.cluster_id, asset_id) in excluded_pairs:
                continue
            representative_id = candidate.representative_asset_id
            if representative_id is None:
                continue
            embedding_id = embedding_id_by_asset.get(representative_id)
            representative_vector = vectors.get(embedding_id) if embedding_id else None
            if representative_vector is None:
                continue
            score = cosine_similarity(vector, representative_vector)
            if score is None or score < thresholds.for_mode(mode):
                continue
            assignment = IncrementalClusterAssignment(
                asset_id=asset_id,
                cluster_id=candidate.cluster_id,
                cluster_mode=mode,
                score=score,
            )
            if best is None or (assignment.score, assignment.cluster_id) > (
                best.score,
                best.cluster_id,
            ):
                best = assignment
        if best is not None:
            return best
    return None


def _candidate_mode(candidate: IncrementalClusterCandidate) -> ClusterMode:
    return (
        candidate.mode
        if isinstance(candidate.mode, ClusterMode)
        else ClusterMode(candidate.mode)
    )


def _embedding_type_value(embedding_type: EmbeddingType | str) -> str:
    return embedding_type.value if isinstance(embedding_type, EmbeddingType) else embedding_type


def _valid_vector(vector: Sequence[float]) -> bool:
    return bool(vector) and all(math.isfinite(value) for value in vector) and any(
        value != 0.0 for value in vector
    )
