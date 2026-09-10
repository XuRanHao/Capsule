"""Shared normalized vector fusion for multimodal search."""

from collections.abc import Sequence

import numpy as np

DEFAULT_NATIVE_CONTENT_WEIGHT = 0.3


def validate_native_content_weight(value: float) -> float:
    """Return a finite native-content weight in the inclusive [0, 1] range."""
    weight = float(value)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("native_content_weight must be between 0 and 1")
    return weight


def fuse_native_dimension_vectors(
    *,
    native_vector: Sequence[float] | None,
    dimension_vector: Sequence[float] | None,
    native_content_weight: float,
) -> list[float]:
    """L2-normalize, blend, and L2-normalize native and dimension vectors."""
    native_weight = validate_native_content_weight(native_content_weight)
    dimension_weight = 1.0 - native_weight
    native = (
        _as_normalized_vector(native_vector, component="native")
        if native_weight > 0.0
        else None
    )
    dimension = (
        _as_normalized_vector(dimension_vector, component="dimension")
        if dimension_weight > 0.0
        else None
    )
    if native is not None and dimension is not None and native.shape != dimension.shape:
        raise ValueError("native and dimension vectors must have the same shape")
    if native is None:
        assert dimension is not None
        fused = dimension_weight * dimension
    elif dimension is None:
        fused = native_weight * native
    else:
        fused = native_weight * native + dimension_weight * dimension
    return [float(value) for value in _l2_normalize(fused)]


def _as_normalized_vector(
    vector: Sequence[float] | None,
    *,
    component: str,
) -> np.ndarray:
    if vector is None:
        raise ValueError(f"{component} vector is required when its weight is non-zero")
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("vectors must be one-dimensional")
    return _l2_normalize(array)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("vector must not be zero-length")
    return vector / norm
