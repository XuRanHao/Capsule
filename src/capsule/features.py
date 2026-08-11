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

FEATURE_DIMENSION_SCOPES: dict[EmbeddingType, str] = {
    EmbeddingType.SUBJECT_CONTENT: (
        "画面或文本中的人物、物体、可识别特征、正在发生的动作、主体关系和内容事实"
    ),
    EmbeddingType.SCENE_THEME: (
        "整幅内容可辨识的叙事语境，包括空间环境、时间与天气、正在发生的事件或活动、题材和"
        "主题情境；缺少整体场景语境的主体陈列或孤立元素不适用"
    ),
    EmbeddingType.VISUAL_STYLE: (
        "素材的视觉表达方式，包括摄影、插画、三维渲染等媒介形态，艺术技法、审美语言、"
        "造型方法、材质表现和渲染质感"
    ),
    EmbeddingType.COLOR_COMPOSITION: (
        "画面的色彩与视觉组织，包括主辅色、冷暖、饱和度、明暗和对比关系，以及光线分布、"
        "视角、景别、画面布局、空间层次和视觉重心"
    ),
    EmbeddingType.MOOD_ATMOSPHERE: (
        "由画面或文本中可核验的光线、色彩、空间、天气、动作、声音和叙事表现共同形成的整体"
        "氛围，包括情绪基调、氛围强度、节奏感和感官体验"
    ),
    EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY: (
        "人物或拟人角色自身的可观察状态，包括表情、身体状态、姿态、神态、互动状态，"
        "以及有明确证据支持的情绪和心理状态"
    ),
    EmbeddingType.ASSET_USAGE: (
        "有具体证据支持的素材用途，包括目标交付物、投放或使用载体、制作任务、工作流环节"
        "和创作参考目的"
    ),
    EmbeddingType.TARGET_AUDIENCE: (
        "素材信息或上下文中明确指向的观看者、使用者、年龄人群、兴趣群体和传播对象"
    ),
    EmbeddingType.PROVENANCE: (
        "素材的来源链路，包括来源平台、数据集或采集渠道，导入与生成方式、派生关系和参考关系"
    ),
    EmbeddingType.RIGHTS_VERSION_AUTHORSHIP: (
        "素材的权利与创作身份信息，包括作者或创作者、所有权、授权范围、版权状态、版本关系和署名要求"
    ),
}


def feature_dimension_scope_prompt() -> str:
    """Render the canonical, positive scope of every model-derived Feature."""

    return "；".join(
        f"{embedding_type.value}={scope}"
        for embedding_type, scope in FEATURE_DIMENSION_SCOPES.items()
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
    resolved_asset_type = asset_type if isinstance(asset_type, AssetType) else AssetType(asset_type)
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
