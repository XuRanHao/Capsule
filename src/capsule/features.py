"""Canonical eligibility rules for Feature-derived Embedding channels."""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from capsule.enums import AssetType, EmbeddingType, FeatureStatus

_NULL_FEATURE_STATUSES = {
    FeatureStatus.UNKNOWN.value,
    FeatureStatus.NOT_APPLICABLE.value,
}

_VISUAL_ASSET_TYPES = frozenset({AssetType.IMAGE, AssetType.VIDEO_SEGMENT})
_VISUAL_ONLY_EMBEDDING_TYPES = frozenset(
    {EmbeddingType.VISUAL_STYLE, EmbeddingType.COLOR_COMPOSITION}
)


def embedding_type_supports_asset_type(
    *,
    embedding_type: EmbeddingType | str,
    asset_type: AssetType | str,
) -> bool:
    """Return whether an embedding channel is meaningful for an Asset type."""

    resolved_embedding_type = (
        embedding_type
        if isinstance(embedding_type, EmbeddingType)
        else EmbeddingType(embedding_type)
    )
    resolved_asset_type = (
        asset_type if isinstance(asset_type, AssetType) else AssetType(asset_type)
    )
    return (
        resolved_embedding_type not in _VISUAL_ONLY_EMBEDDING_TYPES
        or resolved_asset_type in _VISUAL_ASSET_TYPES
    )


def embedding_type_supports_any_asset_type(
    *,
    embedding_type: EmbeddingType | str,
    asset_types: list[AssetType] | tuple[AssetType, ...],
) -> bool:
    """Use union semantics when a search targets multiple Asset types."""

    targets = tuple(asset_types) or tuple(AssetType)
    return any(
        embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=asset_type,
        )
        for asset_type in targets
    )


def effective_feature_text(
    features: Mapping[str, Any],
    embedding_type: EmbeddingType | str,
) -> str | None:
    """Return current Feature text only when the dimension is eligible.

    The POC currently persists the model-oriented ``value`` shape, while the
    specification also defines ``effective_value``/``user_value``/``model_value``.
    This resolver supports both without allowing an explicit null effective value
    or a null status to fall back to stale model text.
    """
    key = embedding_type.value if isinstance(embedding_type, EmbeddingType) else embedding_type
    raw = features.get(key)
    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, Mapping):
        return None

    status = raw.get("status")
    status_value = status.value if isinstance(status, FeatureStatus) else status
    if status_value in _NULL_FEATURE_STATUSES:
        return None

    if "effective_value" in raw:
        return _non_empty_text(raw.get("effective_value"))
    for field in ("user_value", "model_value", "value"):
        value = _non_empty_text(raw.get(field))
        if value is not None:
            return value
    return None


def embedding_channel_is_eligible(
    *,
    embedding_type: EmbeddingType | str,
    asset_features: Mapping[str, Any],
    asset_description: str | None = None,
) -> bool:
    resolved = (
        embedding_type
        if isinstance(embedding_type, EmbeddingType)
        else EmbeddingType(embedding_type)
    )
    if resolved is EmbeddingType.NATIVE_MULTIMODAL:
        return True
    if resolved is EmbeddingType.ASSET_DESCRIPTION:
        return bool(asset_description and asset_description.strip())
    return effective_feature_text(asset_features, resolved) is not None


def asset_usage_embedding_text(
    features: Mapping[str, Any],
    source_relative_path: str,
) -> str | None:
    """Combine usage semantics with a stable directory, excluding unique filenames."""
    usage = effective_feature_text(features, EmbeddingType.ASSET_USAGE)
    if usage is None:
        return None
    normalized = source_relative_path.strip().replace("\\", "/")
    if not normalized:
        return usage
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return usage
    directory = path.parent.as_posix()
    if directory == ".":
        return usage
    return f"素材用途：{usage}；来源目录：{directory}"


def _non_empty_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
