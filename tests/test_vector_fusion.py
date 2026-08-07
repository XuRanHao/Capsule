import numpy as np
import pytest

from capsule.pipeline.vector_fusion import fuse_native_dimension_vectors


def test_fusion_normalizes_components_before_weighted_average() -> None:
    fused = fuse_native_dimension_vectors(
        native_vector=[10.0, 0.0],
        dimension_vector=[0.0, 2.0],
        native_content_weight=0.25,
    )

    assert np.allclose(fused, [0.31622776, 0.94868326])
    assert np.isclose(np.linalg.norm(fused), 1.0)


@pytest.mark.parametrize(
    ("native_vector", "dimension_vector"),
    [([0.0, 0.0], [1.0, 0.0]), ([float("nan"), 0.0], [1.0, 0.0])],
)
def test_fusion_rejects_invalid_component_vectors(
    native_vector: list[float],
    dimension_vector: list[float],
) -> None:
    with pytest.raises(ValueError):
        fuse_native_dimension_vectors(
            native_vector=native_vector,
            dimension_vector=dimension_vector,
            native_content_weight=0.5,
        )


def test_fusion_rejects_a_zero_weighted_result() -> None:
    with pytest.raises(ValueError):
        fuse_native_dimension_vectors(
            native_vector=[1.0, 0.0],
            dimension_vector=[-1.0, 0.0],
            native_content_weight=0.5,
        )


def test_fusion_does_not_require_the_zero_weight_component() -> None:
    dimension_only = fuse_native_dimension_vectors(
        native_vector=None,
        dimension_vector=[0.0, 4.0],
        native_content_weight=0.0,
    )
    native_only = fuse_native_dimension_vectors(
        native_vector=[3.0, 0.0],
        dimension_vector=None,
        native_content_weight=1.0,
    )

    assert np.allclose(dimension_only, [0.0, 1.0])
    assert np.allclose(native_only, [1.0, 0.0])
