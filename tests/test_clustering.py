import numpy as np
import pytest

from capsule.pipeline.clustering import (
    ClusterMemberCandidate,
    HdbscanParameters,
    InsufficientDataError,
    cluster_vectors,
    dataset_hash,
    dynamic_hdbscan_parameters,
    representative_indices,
    select_cluster_representatives,
)


def test_dynamic_hdbscan_parameters() -> None:
    with pytest.raises(InsufficientDataError):
        dynamic_hdbscan_parameters(0)

    assert dynamic_hdbscan_parameters(1).min_cluster_size == 3
    assert dynamic_hdbscan_parameters(1).min_samples == 1
    for sample_count in (2, 14, 15, 50, 200, 1000):
        parameters = dynamic_hdbscan_parameters(sample_count)
        assert parameters.min_cluster_size == 3
        assert parameters.min_samples == 1


def test_cluster_vectors_accepts_a_single_vector() -> None:
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)

    result = cluster_vectors(vectors)

    assert result.labels.tolist() == [-1]
    assert result.noise_count == 1
    assert result.parameter_candidates_evaluated == 1


def test_cluster_vectors_clusters_fewer_than_fifteen_vectors() -> None:
    vectors = np.asarray(
        [
            *[[1.0 + index * 0.001, 0.1 + (index % 2) * 0.001] for index in range(6)],
            *[[0.1 + (index % 2) * 0.001, 1.0 + index * 0.001] for index in range(6)],
        ],
        dtype=np.float32,
    )

    result = cluster_vectors(vectors)

    assert result.parameters == HdbscanParameters(min_cluster_size=3, min_samples=1)
    assert result.cluster_count > 0
    assert result.noise_count == 0


def test_dataset_hash_is_order_independent() -> None:
    assert dataset_hash(["emb_b", "emb_a"]) == dataset_hash(["emb_a", "emb_b"])


def test_representative_indices_excludes_noise() -> None:
    vectors = np.asarray(
        [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, -1], dtype=np.int_)

    representatives = representative_indices(vectors, labels, limit=1)

    assert set(representatives) == {0}
    assert representatives[0][0] in {0, 1}


def test_cluster_vectors_optimization_keeps_fixed_density_parameters() -> None:
    rng = np.random.default_rng(3)
    sample_count = 300
    cluster_count = 4
    cluster_sizes = rng.multinomial(sample_count, [1 / cluster_count] * cluster_count)
    centers = rng.normal(size=(cluster_count, 16))
    centers *= 2.0 / np.linalg.norm(centers, axis=1, keepdims=True)
    vectors = np.vstack(
        [
            centers[label]
            + rng.normal(
                scale=0.3 * (1 + label / (2 * cluster_count)),
                size=(cluster_sizes[label], 16),
            )
            for label in range(cluster_count)
        ]
    ).astype(np.float32)
    vectors = np.hstack(
        [vectors, np.full((sample_count, 48), 0.15, dtype=np.float32)]
    )
    adaptive = cluster_vectors(vectors, optimize_parameters=True)

    assert adaptive.parameter_candidates_evaluated > 1
    assert adaptive.parameters.min_cluster_size == 3
    assert adaptive.parameters.min_samples == 1


def test_cluster_vectors_renormalizes_after_pca() -> None:
    vectors = np.random.default_rng(7).normal(size=(30, 80)).astype(np.float32)

    result = cluster_vectors(
        vectors,
        pca_dimension=12,
        parameters=HdbscanParameters(min_cluster_size=3, min_samples=2),
    )

    projected_norms = np.linalg.norm(result.transformed_vectors, axis=1)
    assert np.allclose(projected_norms, 1.0, atol=1e-6)


def test_cluster_vectors_uses_one_parameter_set_by_default() -> None:
    vectors = np.random.default_rng(9).normal(size=(30, 16)).astype(np.float32)

    result = cluster_vectors(vectors)

    assert result.parameter_candidates_evaluated == 1
    assert result.parameters == dynamic_hdbscan_parameters(len(vectors))


def test_cluster_vectors_rejects_weak_clusters_in_unstructured_data() -> None:
    vectors = np.random.default_rng(0).normal(size=(300, 64)).astype(np.float32)

    result = cluster_vectors(vectors, optimize_parameters=True)

    assert result.cluster_count == 0
    assert result.noise_count == len(vectors)
    assert np.all(result.probabilities == 0.0)


def test_select_cluster_representatives_uses_actual_medoid_and_source_cap() -> None:
    vectors = np.asarray(
        [
            [-2.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0, 0, 0, 0, 0], dtype=np.int_)
    candidates = [
        ClusterMemberCandidate(
            asset_id=f"asset_{index}",
            source_file_id="src_shared" if index in {1, 2, 3} else f"src_{index}",
            membership_probability=0.9 if index < 5 else 0.2 - index * 0.01,
        )
        for index in range(7)
    ]

    selected = select_cluster_representatives(
        vectors,
        labels,
        candidates,
        limit=7,
        edge_limit=2,
    )[0]

    assert selected[0].asset_id == "asset_3"
    assert selected[0].role == "medoid"
    assert selected[0].distance_to_medoid == 0.0
    assert [item.rank for item in selected] == list(range(len(selected)))
    assert sum(item.role == "edge" for item in selected) <= 2
    assert sum(item.source_file_id == "src_shared" for item in selected) <= 2
