import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import hdbscan
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

_MIN_ACCEPTABLE_CLUSTER_QUALITY = 0.05
_SILHOUETTE_SAMPLE_SIZE = 2_000


class InsufficientDataError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class HdbscanParameters:
    min_cluster_size: int
    min_samples: int
    cluster_selection_method: str = "eom"


@dataclass(slots=True)
class ClusterResult:
    labels: NDArray[np.int_]
    probabilities: NDArray[np.float64]
    transformed_vectors: NDArray[np.float32]
    pca_dimension: int
    parameters: HdbscanParameters
    quality_score: float
    parameter_candidates_evaluated: int

    @property
    def cluster_count(self) -> int:
        return len(set(self.labels.tolist()) - {-1})

    @property
    def noise_count(self) -> int:
        return int(np.count_nonzero(self.labels == -1))

    @property
    def noise_ratio(self) -> float:
        return self.noise_count / len(self.labels)


@dataclass(slots=True, frozen=True)
class ClusterMemberCandidate:
    """Metadata for one vector that may become a representative Asset."""

    asset_id: str
    source_file_id: str
    membership_probability: float


@dataclass(slots=True, frozen=True)
class RepresentativeSelection:
    """One ordered Asset selected to explain a cluster rather than the whole cluster."""

    asset_id: str
    source_file_id: str
    role: str
    rank: int
    distance_to_medoid: float
    membership_probability: float


@dataclass(slots=True, frozen=True)
class SemanticMergeParameters:
    """Thresholds for consolidating HDBSCAN microclusters in the source vector space."""

    enabled: bool = True
    centroid_cosine_threshold: float = 0.92
    cross_cluster_mean_cosine_threshold: float = 0.84
    merged_member_min_cosine_threshold: float = 0.92


@dataclass(slots=True, frozen=True)
class SemanticMergeDecision:
    left_label: int
    right_label: int
    target_label: int
    centroid_cosine: float
    cross_cluster_mean_cosine: float
    merged_member_min_cosine: float


@dataclass(slots=True)
class SemanticMergeResult:
    labels: NDArray[np.int_]
    raw_to_merged_labels: dict[int, int]
    decisions: list[SemanticMergeDecision]

    @property
    def raw_cluster_count(self) -> int:
        return len(self.raw_to_merged_labels)

    @property
    def cluster_count(self) -> int:
        return len(set(self.raw_to_merged_labels.values()))


def dynamic_hdbscan_parameters(sample_count: int) -> HdbscanParameters:
    if sample_count < 1:
        raise InsufficientDataError("at least one vector is required")
    return HdbscanParameters(min_cluster_size=3, min_samples=1)


def dataset_hash(embedding_ids: list[str]) -> str:
    canonical = "\n".join(sorted(embedding_ids)).encode()
    return hashlib.sha256(canonical).hexdigest()


def cluster_vectors(
    vectors: NDArray[np.float32],
    *,
    pca_dimension: int | None = 64,
    parameters: HdbscanParameters | None = None,
    optimize_parameters: bool = False,
) -> ClusterResult:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    sample_count, original_dimension = vectors.shape
    base_parameters = parameters or dynamic_hdbscan_parameters(sample_count)

    normalized = _l2_normalize(vectors)
    if sample_count == 1:
        return ClusterResult(
            labels=np.asarray([-1], dtype=np.int_),
            probabilities=np.asarray([0.0], dtype=np.float64),
            transformed_vectors=normalized,
            pca_dimension=original_dimension,
            parameters=base_parameters,
            quality_score=-1.0,
            parameter_candidates_evaluated=1,
        )

    target_dimension = min(
        pca_dimension or original_dimension,
        sample_count - 1,
        original_dimension,
    )
    if target_dimension < original_dimension:
        transformed = PCA(
            n_components=target_dimension,
            random_state=0,
        ).fit_transform(normalized)
    else:
        transformed = normalized

    # PCA centers the vectors and truncation changes their lengths. Normalize
    # again so Euclidean distance continues to represent angular similarity.
    transformed = _l2_normalize_projection(
        np.asarray(transformed, dtype=np.float32),
    )

    parameter_candidates = (
        _hdbscan_parameter_candidates(sample_count, base_parameters)
        if optimize_parameters
        else [base_parameters]
    )
    best = max(
        (_fit_candidate(transformed, candidate) for candidate in parameter_candidates),
        key=lambda candidate: candidate.quality_score,
    )

    # Density clustering can always carve a few tiny groups out of unstructured
    # data. Reject weak adaptive results rather than presenting those groups as
    # meaningful Capsules. An explicitly supplied configuration remains exact.
    if optimize_parameters and best.quality_score < _MIN_ACCEPTABLE_CLUSTER_QUALITY:
        labels = np.full(sample_count, -1, dtype=np.int_)
        probabilities = np.zeros(sample_count, dtype=np.float64)
    else:
        labels = best.labels
        probabilities = best.probabilities

    return ClusterResult(
        labels=np.asarray(labels, dtype=np.int_),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        transformed_vectors=transformed,
        pca_dimension=target_dimension,
        parameters=best.parameters,
        quality_score=best.quality_score,
        parameter_candidates_evaluated=len(parameter_candidates),
    )


def merge_semantically_overlapping_clusters(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int_],
    *,
    parameters: SemanticMergeParameters | None = None,
) -> SemanticMergeResult:
    """Iteratively merge mutually-nearest clusters that pass all semantic gates.

    HDBSCAN remains the source of the raw labels.  Consolidation uses the original
    L2-normalized Embedding space, not the PCA projection, so the configured cosine
    thresholds retain their semantic meaning.  After every accepted merge the
    centroids are recomputed; this prevents a weak A-B-C similarity chain from
    collapsing into one group.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if len(vectors) != len(labels):
        raise ValueError("vectors and labels must have the same length")

    resolved = parameters or SemanticMergeParameters()
    _validate_semantic_merge_parameters(resolved)
    raw_labels = sorted(set(labels.tolist()) - {-1})
    raw_to_merged = {label: label for label in raw_labels}
    if not resolved.enabled or len(raw_labels) < 2:
        return SemanticMergeResult(
            labels=np.asarray(labels, dtype=np.int_).copy(),
            raw_to_merged_labels=raw_to_merged,
            decisions=[],
        )

    normalized = _l2_normalize(vectors)
    groups = {
        label: np.asarray(np.flatnonzero(labels == label), dtype=np.int_) for label in raw_labels
    }
    decisions: list[SemanticMergeDecision] = []

    while len(groups) > 1:
        centroids = {
            label: _normalized_centroid(normalized[indices]) for label, indices in groups.items()
        }
        nearest = _mutual_nearest_labels(centroids)
        candidates: list[SemanticMergeDecision] = []
        for left_label, right_label in nearest:
            left_vectors = normalized[groups[left_label]]
            right_vectors = normalized[groups[right_label]]
            centroid_cosine = float(centroids[left_label] @ centroids[right_label])
            if centroid_cosine < resolved.centroid_cosine_threshold:
                continue

            # Mean of all pairwise cross-cluster cosine values, evaluated without
            # allocating the full |A| x |B| similarity matrix.
            cross_mean = float(left_vectors.mean(axis=0) @ right_vectors.mean(axis=0))
            if cross_mean < resolved.cross_cluster_mean_cosine_threshold:
                continue

            merged_vectors = np.concatenate((left_vectors, right_vectors), axis=0)
            merged_centroid = _normalized_centroid(merged_vectors)
            merged_member_min = float(np.min(merged_vectors @ merged_centroid))
            if merged_member_min < resolved.merged_member_min_cosine_threshold:
                continue

            candidates.append(
                SemanticMergeDecision(
                    left_label=left_label,
                    right_label=right_label,
                    target_label=min(left_label, right_label),
                    centroid_cosine=centroid_cosine,
                    cross_cluster_mean_cosine=cross_mean,
                    merged_member_min_cosine=merged_member_min,
                )
            )

        if not candidates:
            break

        decision = min(
            candidates,
            key=lambda item: (
                -item.centroid_cosine,
                -item.cross_cluster_mean_cosine,
                -item.merged_member_min_cosine,
                item.left_label,
                item.right_label,
            ),
        )
        source_labels = {decision.left_label, decision.right_label}
        groups[decision.target_label] = np.concatenate(
            [groups[label] for label in sorted(source_labels)]
        )
        for label in source_labels - {decision.target_label}:
            del groups[label]
        for raw_label, merged_label in raw_to_merged.items():
            if merged_label in source_labels:
                raw_to_merged[raw_label] = decision.target_label
        decisions.append(decision)

    merged_labels = np.asarray(
        [-1 if int(label) == -1 else raw_to_merged[int(label)] for label in labels],
        dtype=np.int_,
    )
    return SemanticMergeResult(
        labels=merged_labels,
        raw_to_merged_labels=raw_to_merged,
        decisions=decisions,
    )


@dataclass(slots=True)
class _CandidateResult:
    labels: NDArray[np.int_]
    probabilities: NDArray[np.float64]
    parameters: HdbscanParameters
    quality_score: float


def _hdbscan_parameter_candidates(
    sample_count: int,
    base: HdbscanParameters,
) -> list[HdbscanParameters]:
    """Keep density thresholds fixed while comparing cluster selection methods."""
    del sample_count
    alternate_method = "leaf" if base.cluster_selection_method == "eom" else "eom"
    return [
        base,
        HdbscanParameters(
            min_cluster_size=base.min_cluster_size,
            min_samples=base.min_samples,
            cluster_selection_method=alternate_method,
        ),
    ]


def _fit_candidate(
    vectors: NDArray[np.float32],
    parameters: HdbscanParameters,
) -> _CandidateResult:
    model = hdbscan.HDBSCAN(
        min_cluster_size=parameters.min_cluster_size,
        min_samples=parameters.min_samples,
        metric="euclidean",
        cluster_selection_method=parameters.cluster_selection_method,
        allow_single_cluster=False,
        gen_min_span_tree=True,
    )
    labels = np.asarray(model.fit_predict(vectors), dtype=np.int_)
    return _CandidateResult(
        labels=labels,
        probabilities=np.asarray(model.probabilities_, dtype=np.float64),
        parameters=parameters,
        quality_score=_clustering_quality(vectors, labels, model, parameters),
    )


def _clustering_quality(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int_],
    model: hdbscan.HDBSCAN,
    parameters: HdbscanParameters,
) -> float:
    """Score separation, coverage, and resistance to tiny-cluster fragmentation."""
    cluster_labels = set(labels.tolist()) - {-1}
    clustered_mask = labels != -1
    clustered_count = int(np.count_nonzero(clustered_mask))
    if len(cluster_labels) < 2 or clustered_count <= len(cluster_labels):
        return -1.0

    try:
        relative_validity = float(model.relative_validity_)
        silhouette = float(
            silhouette_score(
                vectors[clustered_mask],
                labels[clustered_mask],
                sample_size=min(_SILHOUETTE_SAMPLE_SIZE, clustered_count),
                random_state=0,
            )
        )
    except (AttributeError, ValueError):
        return -1.0
    coverage = clustered_count / len(labels)
    fragmentation = min(
        1.0,
        len(cluster_labels) * parameters.min_cluster_size / clustered_count,
    )
    quality = (
        0.65 * relative_validity
        + 0.35 * silhouette
        - 0.15 * max(0.0, 0.5 - coverage)
        - 0.10 * fragmentation
    )
    return quality if np.isfinite(quality) else -1.0


def representative_indices(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int_],
    *,
    limit: int = 10,
) -> dict[int, list[int]]:
    """Return actual-medoid-adjacent members for callers without Asset metadata."""
    representatives: dict[int, list[int]] = {}
    for label in sorted(set(labels.tolist()) - {-1}):
        member_indices = np.flatnonzero(labels == label)
        members = vectors[member_indices]
        medoid_position = _medoid_position(members)
        medoid = members[medoid_position]
        distances = np.linalg.norm(members - medoid, axis=1)
        order = np.lexsort((member_indices, distances))[:limit]
        representatives[label] = member_indices[order].tolist()
    return representatives


def select_cluster_representatives(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int_],
    candidates: list[ClusterMemberCandidate],
    *,
    limit: int = 10,
    edge_limit: int = 2,
    max_per_source_file: int = 2,
) -> dict[int, list[RepresentativeSelection]]:
    """Select medoid, core and optional edge Assets from each non-noise cluster.

    The vectors must be the PCA-transformed vectors that HDBSCAN clustered.  This
    keeps the medoid and its distances in the same space used to form the cluster.
    No membership threshold is imposed on edge Assets: their lower membership is
    used only for ordering, so the caller can later tune that policy explicitly.
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    if len(vectors) != len(labels) or len(vectors) != len(candidates):
        raise ValueError("vectors, labels, and candidates must have the same length")
    if limit < 1 or edge_limit < 0 or max_per_source_file < 1:
        raise ValueError("representative selection limits must be positive")

    selections: dict[int, list[RepresentativeSelection]] = {}
    for label in sorted(set(labels.tolist()) - {-1}):
        member_indices = np.flatnonzero(labels == label)
        members = vectors[member_indices]
        medoid_index = int(member_indices[_medoid_position(members)])
        medoid_vector = vectors[medoid_index]
        distances = {
            int(index): float(np.linalg.norm(vectors[index] - medoid_vector))
            for index in member_indices
        }
        target_count = min(limit, len(member_indices))
        planned_edge_count = min(edge_limit, max(0, target_count - 5))
        core_target = target_count - planned_edge_count

        selected_indices = [medoid_index]
        source_counts = {candidates[medoid_index].source_file_id: 1}

        closest_first = sorted(
            (int(index) for index in member_indices if int(index) != medoid_index),
            key=lambda index: (distances[index], candidates[index].asset_id),
        )
        _append_with_source_cap(
            selected_indices,
            closest_first,
            candidates,
            source_counts,
            max_per_source_file=max_per_source_file,
            maximum=core_target,
        )

        edge_indices: list[int] = []
        edge_candidates = sorted(
            (index for index in closest_first if index not in selected_indices),
            key=lambda index: (
                candidates[index].membership_probability,
                distances[index],
                candidates[index].asset_id,
            ),
        )
        _append_with_source_cap(
            edge_indices,
            edge_candidates,
            candidates,
            source_counts,
            max_per_source_file=max_per_source_file,
            maximum=planned_edge_count,
        )

        # If a source-file cap prevented the planned core/edge balance, fill any
        # remaining slots with the closest still-eligible members.
        _append_with_source_cap(
            selected_indices,
            (
                index
                for index in closest_first
                if index not in selected_indices and index not in edge_indices
            ),
            candidates,
            source_counts,
            max_per_source_file=max_per_source_file,
            maximum=target_count - len(edge_indices),
        )
        ordered_indices = [*selected_indices, *edge_indices]
        selections[label] = [
            RepresentativeSelection(
                asset_id=candidates[index].asset_id,
                source_file_id=candidates[index].source_file_id,
                role="medoid"
                if index == medoid_index
                else "edge"
                if index in edge_indices
                else "core",
                rank=rank,
                distance_to_medoid=distances[index],
                membership_probability=candidates[index].membership_probability,
            )
            for rank, index in enumerate(ordered_indices)
        ]
    return selections


def _medoid_position(members: NDArray[np.float32]) -> int:
    distances = np.linalg.norm(members[:, np.newaxis, :] - members[np.newaxis, :, :], axis=2)
    total_distances = distances.sum(axis=1)
    return int(np.argmin(total_distances))


def _append_with_source_cap(
    selected: list[int],
    candidates_to_add: Iterable[int],
    candidates: list[ClusterMemberCandidate],
    source_counts: dict[str, int],
    *,
    max_per_source_file: int,
    maximum: int,
) -> None:
    for index in candidates_to_add:
        if len(selected) >= maximum:
            return
        source_file_id = candidates[index].source_file_id
        if source_counts.get(source_file_id, 0) >= max_per_source_file:
            continue
        selected.append(index)
        source_counts[source_file_id] = source_counts.get(source_file_id, 0) + 1


def _l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("vectors contain an all-zero row")
    return np.asarray(vectors / norms, dtype=np.float32)


def _l2_normalize_projection(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize projected rows while allowing a point exactly at the PCA origin."""
    if not np.isfinite(vectors).all():
        raise ValueError("PCA projection contains NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > np.finfo(np.float32).eps, norms, 1.0)
    return np.asarray(vectors / safe_norms, dtype=np.float32)


def _validate_semantic_merge_parameters(parameters: SemanticMergeParameters) -> None:
    thresholds = (
        parameters.centroid_cosine_threshold,
        parameters.cross_cluster_mean_cosine_threshold,
        parameters.merged_member_min_cosine_threshold,
    )
    if any(not np.isfinite(value) or value < -1.0 or value > 1.0 for value in thresholds):
        raise ValueError("semantic merge cosine thresholds must be between -1 and 1")


def _normalized_centroid(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    centroid = np.asarray(vectors.mean(axis=0), dtype=np.float32)
    norm = float(np.linalg.norm(centroid))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("semantic cluster centroid has zero length")
    return np.asarray(centroid / norm, dtype=np.float32)


def _mutual_nearest_labels(
    centroids: dict[int, NDArray[np.float32]],
) -> list[tuple[int, int]]:
    nearest: dict[int, int] = {}
    for label in sorted(centroids):
        others = [candidate for candidate in sorted(centroids) if candidate != label]
        nearest[label] = min(
            others,
            key=lambda candidate: (
                -float(centroids[label] @ centroids[candidate]),
                candidate,
            ),
        )
    return [
        (left, right)
        for left, right in sorted(nearest.items())
        if left < right and nearest.get(right) == left
    ]
