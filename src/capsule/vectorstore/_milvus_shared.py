"""Small domain-neutral helpers shared by Capsule Milvus collections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def validate_vector(vector: Sequence[float], *, dimension: int) -> None:
    """Reject vectors that cannot be safely indexed or searched."""

    if len(vector) != dimension:
        raise ValueError(f"expected {dimension} dimensions, got {len(vector)}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("vector contains NaN or infinity")
    if not any(value != 0.0 for value in vector):
        raise ValueError("vector must not be all zeros")


def field_dimension(field: object) -> int | None:
    """Read a Milvus vector field dimension from a client description payload."""

    if not isinstance(field, dict):
        return None
    params = field.get("params")
    value = params.get("dim") if isinstance(params, dict) else field.get("dim")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def delete_count(response: Any) -> int:
    """Normalize the delete response returned by supported Milvus client versions."""

    if not isinstance(response, dict):
        return 0
    value = response.get("delete_count", response.get("delete_cnt", 0))
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
