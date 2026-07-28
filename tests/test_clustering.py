import numpy as np
import pytest

from capsule.pipeline.clustering import (
    InsufficientDataError,
    dataset_hash,
    dynamic_hdbscan_parameters,
    representative_indices,
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
