"""Canonical eligibility rules for Feature-derived Embedding channels."""

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from capsule.enums import AssetType, EmbeddingType, FeatureApplicability, FeatureSalience

_NULL_FEATURE_APPLICABILITY = {
    FeatureApplicability.UNKNOWN.value,
    FeatureApplicability.NOT_APPLICABLE.value,
}

_VISUAL_ASSET_TYPES = frozenset({AssetType.IMAGE, AssetType.VIDEO_SEGMENT})
ACTIVE_EMBEDDING_TYPES: tuple[EmbeddingType, ...] = (
    EmbeddingType.NATIVE_MULTIMODAL,
    EmbeddingType.SUBJECT_CONTENT,
    EmbeddingType.SCENE_THEME,
    EmbeddingType.VISUAL_PRESENTATION,
)

FEATURE_EMBEDDING_TYPES: tuple[EmbeddingType, ...] = (
    EmbeddingType.SUBJECT_CONTENT,
    EmbeddingType.SCENE_THEME,
    EmbeddingType.VISUAL_PRESENTATION,
)

_VISUAL_ONLY_EMBEDDING_TYPES = frozenset(
    {
        EmbeddingType.VISUAL_PRESENTATION,
        # Historical channels remain readable for existing Assets.
        EmbeddingType.VISUAL_STYLE,
        EmbeddingType.COLOR_COMPOSITION,
    }
)

FEATURE_DIMENSION_SCOPES: dict[EmbeddingType, str] = {
    EmbeddingType.SUBJECT_CONTENT: (
        "聚焦素材中可被指认的核心对象及其内容事实：人物、动物、物体的身份或类别，具有辨识度"
        "的外观与部件、姿态动作、持有物，以及对象之间的互动和关系。描述应帮助检索者在不同"
        "背景和视觉风格中找到同类主体或同类内容；必要时可带最短上下文来说明动作和关系"
    ),
    EmbeddingType.SCENE_THEME: (
        "聚焦整幅素材所呈现的全局情境：空间环境、时间、天气、正在发生的整体事件或活动、叙事"
        "语境和题材主题。描述应帮助检索者在具体人物或物体不同的情况下找到同类场所、活动或"
        "故事情境；可保留理解情境所需的泛化参与者。角色三视图、产品白底陈列、透明背景元素、"
        "局部特写或其他不能形成独立全局情境的孤立展示应设 "
        "applicability=not_applicable"
    ),
    EmbeddingType.VISUAL_PRESENTATION: (
        "聚焦素材如何被视觉化呈现：摄影、插画、三维渲染等媒介与成像方式，写实或风格化程度，"
        "线条、笔触、形体概括、材质和渲染质感；主辅色、冷暖、饱和度、明度、对比及色彩分布；"
        "光线方向与软硬、明暗层次、阴影和高光；视角、景别、透视、留白、均衡、节奏、视觉"
        "重心、前中后景和空间深度。描述应帮助检索者跨主体、跨场景找到呈现方式相近的素材；"
        "可用主体、背景、前景、点缀等抽象画面角色明确视觉属性的作用位置"
    ),
}


FEATURE_DIMENSION_DISAMBIGUATION_PROMPT = (
    "请把同一素材理解为三个互补的检索视角，并分别完成三次聚焦：subject_content 聚焦“素材"
    "里有哪些核心对象、它们是什么状态以及彼此有什么关系”；scene_theme 聚焦“整幅素材处于"
    "什么环境、时间、活动或故事情境”；visual_presentation 聚焦“这些内容通过什么媒介、"
    "造型、色光和构图方式呈现”。每个维度可以保留帮助理解的最短上下文，但主要信息必须服务"
    "于该维度的检索目标。例如同一张图可分别写“持剑少女与对手对峙”（主体）、“雪夜街道中的"
    "战斗场面”（场景）、“冷蓝水彩、逆光剪影、中央构图”（视觉）。少量反例：不要把“冷蓝"
    "逆光”作为主体特征；不要把“少女的银色长发”作为场景主题；不要把“风不觉持剑”作为"
    "视觉表现。不要把一条完整素材描述原样复制到三个维度。"
)


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
    """Return one salience-structured text for a Feature dimension.

    New records store independently attributable ``items``. Historical records
    using ``value`` or override fields remain readable while they are gradually
    regenerated.
    """
    key = embedding_type.value if isinstance(embedding_type, EmbeddingType) else embedding_type
    raw = features.get(key)
    if raw is None and key == EmbeddingType.VISUAL_PRESENTATION.value:
        legacy_parts = [
            _effective_raw_feature_text(features.get(legacy_key))
            for legacy_key in (
                EmbeddingType.VISUAL_STYLE.value,
                EmbeddingType.COLOR_COMPOSITION.value,
            )
        ]
        combined = "；".join(part for part in legacy_parts if part)
        return combined or None
    return _effective_raw_feature_text(raw)


def _effective_raw_feature_text(raw: object) -> str | None:
    """Read one current or historical Feature payload."""

    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, Mapping):
        return None

    applicability = raw.get("applicability")
    applicability_value = (
        applicability.value
        if isinstance(applicability, FeatureApplicability)
        else applicability
    )
    if applicability_value in _NULL_FEATURE_APPLICABILITY:
        return None

    items = raw.get("items")
    if isinstance(items, list):
        grouped: dict[str, list[str]] = {
            FeatureSalience.HIGH.value: [],
            FeatureSalience.MEDIUM.value: [],
            FeatureSalience.LOW.value: [],
        }
        for item in items[:5]:
            if not isinstance(item, Mapping):
                continue
            description = _non_empty_text(item.get("description"))
            if description is None:
                continue
            subject = _non_empty_text(item.get("subject"))
            rendered_item = (
                f"{subject}：{description}"
                if subject is not None and subject.casefold() != description.casefold()
                else description
            )
            raw_salience = item.get("salience")
            if isinstance(raw_salience, FeatureSalience):
                salience_key = raw_salience.value
            elif isinstance(raw_salience, (int, float)) and not isinstance(
                raw_salience, bool
            ):
                salience_key = (
                    FeatureSalience.HIGH.value
                    if raw_salience >= 0.67
                    else (
                        FeatureSalience.MEDIUM.value
                        if raw_salience >= 0.34
                        else FeatureSalience.LOW.value
                    )
                )
            elif isinstance(raw_salience, str) and raw_salience in grouped:
                salience_key = raw_salience
            else:
                salience_key = FeatureSalience.MEDIUM.value
            if rendered_item not in grouped[salience_key]:
                grouped[salience_key].append(rendered_item)
        sections = [
            ("核心信息", grouped[FeatureSalience.HIGH.value]),
            ("重要补充", grouped[FeatureSalience.MEDIUM.value]),
            ("辅助信息", grouped[FeatureSalience.LOW.value]),
        ]
        rendered = "；".join(
            f"{label}：{'；'.join(descriptions)}"
            for label, descriptions in sections
            if descriptions
        )
        return rendered or None

    status = raw.get("status")
    if status in _NULL_FEATURE_APPLICABILITY:
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
