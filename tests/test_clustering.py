import numpy as np
import pytest

from capsule.pipeline.clustering import (
    ClusterMemberCandidate,
    InsufficientDataError,
    dataset_hash,
    dynamic_hdbscan_parameters,
    representative_indices,
    select_cluster_representatives,
)


def test_dynamic_hdbscan_parameters() -> None:
    with pytest.raises(InsufficientDataError):
        dynamic_hdbscan_parameters(14)

    assert dynamic_hdbscan_parameters(15).min_cluster_size == 3
    assert dynamic_hdbscan_parameters(50).min_cluster_size == 5
    assert dynamic_hdbscan_parameters(200).min_cluster_size == 10
    assert dynamic_hdbscan_parameters(1000).min_cluster_size == 20


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
