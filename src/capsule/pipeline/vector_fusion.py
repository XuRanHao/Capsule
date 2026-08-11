"""Backward-compatible import path for the shared vector fusion helpers."""

from capsule.vector_fusion import (
    DEFAULT_NATIVE_CONTENT_WEIGHT,
    fuse_native_dimension_vectors,
    validate_native_content_weight,
)

__all__ = [
    "DEFAULT_NATIVE_CONTENT_WEIGHT",
    "fuse_native_dimension_vectors",
    "validate_native_content_weight",
]
