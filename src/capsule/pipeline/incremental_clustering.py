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
from typing import Any, Protocol

from capsule.config import Settings
from capsule.db.repositories import ClusterBootstrapState
from capsule.enums import ClusterMemberSource, ClusterMode, EmbeddingType
from capsule.pipeline.vector_fusion import fuse_native_dimension_vectors

logger = logging.getLogger(__name__)


class IncrementalClusterCandidate(Protocol):
    @property
    def cluster_id(self) -> str: ...

    @property
    def mode(self) -> ClusterMode | str: ...

    @property
    def representative_asset_id(self) -> str | None: ...

    @property
    def native_content_weight(self) -> float: ...


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


class ClusterBootstrapRepository(Protocol):
    async def get_cluster_bootstrap_state(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> ClusterBootstrapState: ...


class FullClusterRunner(Protocol):
    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL,
        cluster_run_id: str | None = None,
        pca_dimension: int = 8,
        min_samples: int = 3,
        min_cluster_size: int = 2,
        optimize_parameters: bool = False,
        trigger: str = "user",
    ) -> object: ...


class IncrementalRelationGraphUpdater(Protocol):
    async def update_assets(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
        affected_cluster_ids: Sequence[str],
    ) -> Any: ...

    async def build(
        self,
        *,
        workspace_id: str,
    ) -> dict[str, Any]: ...


@dataclass(slots=True, frozen=True)
class IncrementalAssignmentThresholds:
    """Cosine thresholds for the two algorithm-addressable cluster modes."""

    resident_open: float = 0.88
    dynamic: float = 0.88

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
class ClusterBootstrapDecision:
    workspace_id: str
    embedding_type: EmbeddingType
    has_baseline: bool
    run_in_progress: bool
    eligible_asset_count: int
    minimum_asset_count: int
    should_bootstrap: bool
    new_asset_count: int = 0
    new_asset_ratio: float = 0.0
    should_recluster: bool = False


@dataclass(slots=True, frozen=True)
class IncrementalClusterProcessResult:
    assignment: IncrementalAssignmentResult
    bootstrap: ClusterBootstrapDecision
    bootstrap_scheduled: bool


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
        all_asset_ids = (*requested_asset_ids, *representative_asset_ids)
        dimension_embeddings = await self._repository.list_indexed_asset_embeddings(
            workspace_id=workspace_id,
            embedding_type=embedding_type_value,
            asset_ids=all_asset_ids,
        )
        dimension_embedding_id_by_asset = {
            item.asset_id: item.embedding_id for item in dimension_embeddings
        }

        native_only = embedding_type is EmbeddingType.NATIVE_MULTIMODAL
        candidate_weights = {
            _candidate_native_content_weight(candidate, native_only=native_only)
            for candidate in candidates
        }
        need_native_vectors = not native_only and any(weight > 0.0 for weight in candidate_weights)
        native_embedding_id_by_asset = dimension_embedding_id_by_asset
        if need_native_vectors:
            native_embeddings = await self._repository.list_indexed_asset_embeddings(
                workspace_id=workspace_id,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
                asset_ids=all_asset_ids,
            )
            native_embedding_id_by_asset = {
                item.asset_id: item.embedding_id for item in native_embeddings
            }

        requested_embedding_ids = set(dimension_embedding_id_by_asset.values())
        if need_native_vectors:
            requested_embedding_ids.update(native_embedding_id_by_asset.values())
        vectors: dict[str, list[float]] = {}
        if requested_embedding_ids:
            await self._vector_store.ensure_collection()
            vectors = await self._vector_store.fetch_vectors(sorted(requested_embedding_ids))

        weights_to_build = candidate_weights or {0.0}
        vectors_by_asset_and_weight: dict[tuple[str, float], list[float]] = {}
        for asset_id in all_asset_ids:
            for native_content_weight in weights_to_build:
                vector = _fused_asset_vector(
                    asset_id=asset_id,
                    embedding_type=embedding_type,
                    native_content_weight=native_content_weight,
                    dimension_embedding_id_by_asset=dimension_embedding_id_by_asset,
                    native_embedding_id_by_asset=native_embedding_id_by_asset,
                    vectors=vectors,
                )
                if vector is not None:
                    vectors_by_asset_and_weight[(asset_id, native_content_weight)] = vector

        missing_vector_assets = {
            asset_id
            for asset_id in requested_asset_ids
            if not any(
                (asset_id, native_content_weight) in vectors_by_asset_and_weight
                for native_content_weight in weights_to_build
            )
        }
        usable_candidates = [
            candidate
            for candidate in candidates
            if candidate.representative_asset_id is not None
            and (
                candidate.representative_asset_id,
                _candidate_native_content_weight(candidate, native_only=native_only),
            )
            in vectors_by_asset_and_weight
        ]
        target_asset_ids = tuple(
            asset_id for asset_id in requested_asset_ids if asset_id not in missing_vector_assets
        )
        excluded_pairs = await self._repository.list_excluded_pairs(
            workspace_id=workspace_id,
            embedding_type=embedding_type_value,
            cluster_ids=tuple(candidate.cluster_id for candidate in usable_candidates),
            asset_ids=target_asset_ids,
        )
        thresholds = self._thresholds_by_embedding_type.get(
            embedding_type_value,
            self._default_thresholds,
        )

        planned: list[IncrementalClusterAssignment] = []
        for asset_id in target_asset_ids:
            assignment = _select_assignment(
                asset_id=asset_id,
                candidates=usable_candidates,
                vectors_by_asset_and_weight=vectors_by_asset_and_weight,
                native_only=native_only,
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
    """Assign new assets and schedule only the first baseline clustering run."""

    def __init__(
        self,
        *,
        settings: Settings,
        assignment_service: IncrementalClusterService,
        repository: ClusterBootstrapRepository,
        cluster_runner: FullClusterRunner,
        relation_graph_updater: IncrementalRelationGraphUpdater | None = None,
    ) -> None:
        self._settings = settings
        self._assignment_service = assignment_service
        self._repository = repository
        self._cluster_runner = cluster_runner
        self._relation_graph_updater = relation_graph_updater
        self._semaphore = asyncio.Semaphore(settings.cluster_bootstrap_concurrency)
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
        state = await self._repository.get_cluster_bootstrap_state(
            workspace_id=workspace_id,
            embedding_type=embedding_type.value,
            model_name=self._settings.embedding_model,
            dimension=self._settings.embedding_dimension,
            milvus_collection=self._settings.milvus_collection,
        )
        decision = evaluate_cluster_bootstrap(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            state=state,
            minimum_asset_count=self._settings.cluster_bootstrap_minimum_count,
            auto_recluster_new_ratio=self._settings.cluster_auto_recluster_new_ratio,
            auto_recluster_minimum_new_count=(
                self._settings.cluster_auto_recluster_minimum_new_count
            ),
        )
        if (
            embedding_type is EmbeddingType.SUBJECT_CONTENT
            and self._relation_graph_updater is not None
        ):
            try:
                await self._relation_graph_updater.update_assets(
                    workspace_id=workspace_id,
                    asset_ids=asset_ids,
                    affected_cluster_ids=tuple(
                        dict.fromkeys(item.cluster_id for item in assignment.assignments)
                    ),
                )
            except Exception:
                logger.exception(
                    "incremental relationship update failed for workspace=%s",
                    workspace_id,
                )
        key = (workspace_id, embedding_type)
        scheduled = (
            decision.should_bootstrap
            or (
                embedding_type is EmbeddingType.SUBJECT_CONTENT
                and decision.should_recluster
            )
        ) and key not in self._running_keys
        if scheduled:
            self._running_keys.add(key)
            trigger = (
                "automatic_bootstrap"
                if decision.should_bootstrap
                else "automatic_recluster"
            )
            task = asyncio.create_task(self._run_full(key, trigger=trigger))
            self._tasks.add(task)
            task.add_done_callback(self._bootstrap_done)
        return IncrementalClusterProcessResult(
            assignment=assignment,
            bootstrap=decision,
            bootstrap_scheduled=scheduled,
        )

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_full(
        self,
        key: tuple[str, EmbeddingType],
        *,
        trigger: str,
    ) -> None:
        workspace_id, embedding_type = key
        try:
            async with self._semaphore:
                await self._cluster_runner.run(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type,
                    trigger=trigger,
                )
        finally:
            self._running_keys.discard(key)

    def _bootstrap_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except Exception:
            logger.exception("background dimension bootstrap clustering failed")


def evaluate_cluster_bootstrap(
    *,
    workspace_id: str,
    embedding_type: EmbeddingType,
    state: ClusterBootstrapState,
    minimum_asset_count: int,
    auto_recluster_new_ratio: float = 1.0,
    auto_recluster_minimum_new_count: int = 2**31 - 1,
) -> ClusterBootstrapDecision:
    """Decide whether one dimension needs its first automatic baseline run."""

    if minimum_asset_count < 1:
        raise ValueError("minimum_asset_count must be at least 1")
    if state.eligible_asset_count < 0:
        raise ValueError("eligible_asset_count cannot be negative")
    new_asset_count = state.new_asset_count
    new_asset_ratio = (
        new_asset_count / state.eligible_asset_count
        if state.eligible_asset_count
        else 0.0
    )
    return ClusterBootstrapDecision(
        workspace_id=workspace_id,
        embedding_type=embedding_type,
        has_baseline=state.has_baseline,
        run_in_progress=state.run_in_progress,
        eligible_asset_count=state.eligible_asset_count,
        minimum_asset_count=minimum_asset_count,
        should_bootstrap=(
            not state.has_baseline
            and not state.run_in_progress
            and state.eligible_asset_count >= minimum_asset_count
        ),
        new_asset_count=new_asset_count,
        new_asset_ratio=new_asset_ratio,
        should_recluster=(
            state.has_baseline
            and not state.run_in_progress
            and new_asset_count >= auto_recluster_minimum_new_count
            and new_asset_ratio >= auto_recluster_new_ratio
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
    candidates: Sequence[IncrementalClusterCandidate],
    vectors_by_asset_and_weight: Mapping[tuple[str, float], list[float]],
    native_only: bool,
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
            native_content_weight = _candidate_native_content_weight(
                candidate,
                native_only=native_only,
            )
            target_vector = vectors_by_asset_and_weight.get((asset_id, native_content_weight))
            representative_vector = vectors_by_asset_and_weight.get(
                (representative_id, native_content_weight)
            )
            if target_vector is None or representative_vector is None:
                continue
            score = cosine_similarity(target_vector, representative_vector)
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


def _candidate_native_content_weight(
    candidate: IncrementalClusterCandidate,
    *,
    native_only: bool,
) -> float:
    """Return the persisted fusion setting, keeping legacy runs dimension-only."""
    if native_only:
        return 1.0
    value = getattr(candidate, "native_content_weight", 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 0.0
    return weight if math.isfinite(weight) and 0.0 <= weight <= 1.0 else 0.0


def _fused_asset_vector(
    *,
    asset_id: str,
    embedding_type: EmbeddingType,
    native_content_weight: float,
    dimension_embedding_id_by_asset: Mapping[str, str],
    native_embedding_id_by_asset: Mapping[str, str],
    vectors: Mapping[str, list[float]],
) -> list[float] | None:
    """Build one Asset representation compatible with a candidate's full run."""
    dimension_embedding_id = dimension_embedding_id_by_asset.get(asset_id)
    if dimension_embedding_id is None:
        return None
    dimension_vector = vectors.get(dimension_embedding_id)
    if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
        if dimension_vector is not None and _valid_vector(dimension_vector):
            return dimension_vector
        return None

    native_vector: Sequence[float] | None = None
    if native_content_weight > 0.0:
        native_embedding_id = native_embedding_id_by_asset.get(asset_id)
        native_vector = vectors.get(native_embedding_id) if native_embedding_id else None
    if native_content_weight < 1.0 and (
        dimension_vector is None or not _valid_vector(dimension_vector)
    ):
        return None
    if native_content_weight > 0.0 and (native_vector is None or not _valid_vector(native_vector)):
        return None
    try:
        return fuse_native_dimension_vectors(
            native_vector=native_vector,
            dimension_vector=dimension_vector,
            native_content_weight=native_content_weight,
        )
    except ValueError:
        return None
