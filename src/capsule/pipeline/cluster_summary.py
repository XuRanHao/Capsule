"""Build the deliberately small model input used to name one Cluster Capsule."""

import json
from dataclasses import dataclass
from typing import Any

from capsule.enums import ClusterRepresentativeRole

EMBEDDING_DIMENSION_LABELS: dict[str, str] = {
    "native_multimodal": "跨模态内容语义",
    "asset_description": "资产自然语言描述",
    "subject_content": "主体与内容",
    "scene_theme": "场景与题材",
    "visual_style": "视觉风格",
    "color_composition": "色彩与构图",
    "mood_atmosphere": "画面情绪氛围",
    "character_state_or_psychology": "人物状态或心理",
    "asset_usage": "资产用途",
    "target_audience": "目标受众",
    "provenance": "来源与创作关系",
    "rights_version_authorship": "权利、版本与作者",
}


@dataclass(slots=True, frozen=True)
class ClusterSummaryRepresentative:
    """The selected Asset fields that are safe and useful for cluster naming."""

    asset_id: str
    role: ClusterRepresentativeRole
    asset_type: str
    asset_name: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    file_tree_context: list[str]
    membership_probability: float
    distance_to_medoid: float


def build_cluster_summary_messages(
    *,
    embedding_type: str,
    member_count: int,
    average_membership_probability: float,
    representatives: list[ClusterSummaryRepresentative],
) -> list[dict[str, str]]:
    """Create a prompt containing *only* selected representative Assets.

    The full cluster is intentionally excluded.  Representative rows include their
    persisted Asset IDs so the returned capsule can be traced back without using a
    description as an identifier.
    """
    if not representatives:
        raise ValueError("a Cluster Capsule summary requires representative Assets")

    dimension_label = EMBEDDING_DIMENSION_LABELS.get(embedding_type, embedding_type)
    payload = {
        "embedding_type": embedding_type,
        "semantic_dimension": dimension_label,
        "cluster_statistics": {
            "member_count": member_count,
            "average_membership_probability": round(average_membership_probability, 4),
        },
        "representative_assets": [
            {
                "asset_id": asset.asset_id,
                "role": asset.role.value,
                "asset_type": asset.asset_type,
                "asset_name": asset.asset_name,
                "asset_description": asset.asset_description,
                "current_dimension_feature": _effective_feature(
                    asset.asset_features.get(embedding_type)
                ),
                "file_tree_context": asset.file_tree_context,
                "membership_probability": round(asset.membership_probability, 4),
                "distance_to_medoid": round(asset.distance_to_medoid, 6),
            }
            for asset in representatives
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "你正在总结一个通过向量聚类发现的个人资产组。只依据输入中的代表资产，"
                "不要猜测或补充作者、版权、项目、人物身份。名称必须反映 semantic_dimension，"
                "并使用简洁、可区分的中文。description 使用中文，50 到 150 字；如果代表资产"
                "之间差异明显，要在描述中说明。只返回合法 JSON，格式为："
                '{"name":"...","description":"...","keywords":["..."],'
                '"common_features":["..."],"internal_variance":"low|medium|high"}。'
                "keywords 必须有 3 到 8 个。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _effective_feature(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for field in ("effective_value", "user_value", "model_value", "value"):
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None
