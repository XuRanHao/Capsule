import hashlib
from dataclasses import dataclass

import hdbscan
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


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

    @property
    def cluster_count(self) -> int:
        return len(set(self.labels.tolist()) - {-1})

    @property
    def noise_count(self) -> int:
        return int(np.count_nonzero(self.labels == -1))

    @property
    def noise_ratio(self) -> float:
        return self.noise_count / len(self.labels)


def dynamic_hdbscan_parameters(sample_count: int) -> HdbscanParameters:
    if sample_count < 15:
        raise InsufficientDataError("at least 15 vectors are required")
    if sample_count < 50:
        return HdbscanParameters(min_cluster_size=3, min_samples=2)
    if sample_count < 200:
        return HdbscanParameters(min_cluster_size=5, min_samples=3)
    if sample_count < 1000:
        return HdbscanParameters(min_cluster_size=10, min_samples=5)
    return HdbscanParameters(min_cluster_size=20, min_samples=10)


def dataset_hash(embedding_ids: list[str]) -> str:
    canonical = "\n".join(sorted(embedding_ids)).encode()
    return hashlib.sha256(canonical).hexdigest()


def cluster_vectors(
    vectors: NDArray[np.float32],
    *,
    pca_dimension: int | None = 64,
    parameters: HdbscanParameters | None = None,
) -> ClusterResult:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional matrix")
    sample_count, original_dimension = vectors.shape
    parameters = parameters or dynamic_hdbscan_parameters(sample_count)

    normalized = _l2_normalize(vectors)
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

    transformed = np.asarray(transformed, dtype=np.float32)
    model = hdbscan.HDBSCAN(
        min_cluster_size=parameters.min_cluster_size,
        min_samples=parameters.min_samples,
        metric="euclidean",
        cluster_selection_method=parameters.cluster_selection_method,
        allow_single_cluster=False,
        prediction_data=True,
    )
    labels = model.fit_predict(transformed)
    return ClusterResult(
        labels=np.asarray(labels, dtype=np.int_),
        probabilities=np.asarray(model.probabilities_, dtype=np.float64),
        transformed_vectors=transformed,
        pca_dimension=target_dimension,
        parameters=parameters,
    )


def representative_indices(
    vectors: NDArray[np.float32],
    labels: NDArray[np.int_],
    *,
    limit: int = 10,
) -> dict[int, list[int]]:
    representatives: dict[int, list[int]] = {}
    for label in sorted(set(labels.tolist()) - {-1}):
        member_indices = np.flatnonzero(labels == label)
        members = vectors[member_indices]
        centroid = members.mean(axis=0)
        distances = np.linalg.norm(members - centroid, axis=1)
        order = np.argsort(distances)[:limit]
        representatives[label] = member_indices[order].tolist()
    return representatives


def _l2_normalize(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if not np.isfinite(vectors).all():
        raise ValueError("vectors contain NaN or infinity")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("vectors contain an all-zero row")
    return np.asarray(vectors / norms, dtype=np.float32)
