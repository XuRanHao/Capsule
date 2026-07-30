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

_GENERIC_TITLE_TERMS = (
    "图像",
    "图片",
    "素材",
    "作品",
    "文件",
    "聚类",
    "分组",
    "集合",
    "类别",
)


@dataclass(slots=True, frozen=True)
class ClusterSummaryDimensionPolicy:
    """The semantic boundary used to describe and name one embedding channel."""

    description_focus: str
    title_focus: str
    description_must_exclude: tuple[str, ...]
    title_must_exclude: tuple[str, ...] = _GENERIC_TITLE_TERMS


CLUSTER_SUMMARY_DIMENSION_POLICIES: dict[str, ClusterSummaryDimensionPolicy] = {
    "native_multimodal": ClusterSummaryDimensionPolicy(
        description_focus="跨模态内容中共同出现的核心语义、对象关系、行为和上下文",
        title_focus="最能区分该簇的核心内容语义",
        description_must_exclude=("无证据的作者、版权、来源、版本和创作关系",),
    ),
    "asset_description": ClusterSummaryDimensionPolicy(
        description_focus="资产自然语言描述中反复出现的事实、对象、行为和语义关系",
        title_focus="自然语言描述中的核心共同语义",
        description_must_exclude=("描述中没有明确出现的推断信息",),
    ),
    "subject_content": ClusterSummaryDimensionPolicy(
        description_focus="主体对象、人物或物体特征、动作以及主体之间的关系",
        title_focus="核心主体或主体动作",
        description_must_exclude=(
            "场景题材",
            "视觉风格",
            "颜色构图",
            "情绪氛围",
            "媒介形式",
            "资产用途",
            "目标受众",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "场景", "风格", "配色", "氛围", "用途"),
    ),
    "scene_theme": ClusterSummaryDimensionPolicy(
        description_focus="空间环境、时间天气、活动情境以及画面题材或主题",
        title_focus="核心场景或题材",
        description_must_exclude=(
            "视觉风格",
            "颜色构图",
            "媒介形式",
            "资产用途",
            "目标受众",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "风格", "配色", "构图", "受众"),
    ),
    "visual_style": ClusterSummaryDimensionPolicy(
        description_focus="表现技法、视觉语言、审美流派、渲染质感和风格化程度",
        title_focus="最有区分度的视觉风格或表现技法",
        description_must_exclude=(
            "具体人物身份",
            "具体主体内容",
            "场景题材",
            "资产用途",
            "目标受众",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "主题", "场景", "宣传", "用途"),
    ),
    "color_composition": ClusterSummaryDimensionPolicy(
        description_focus="主辅色、冷暖、明暗、饱和度、对比度、色彩关系和构图层次",
        title_focus="只提炼颜色、冷暖、明暗、饱和度、对比度或撞色关系",
        description_must_exclude=(
            "人物或物体身份",
            "场景题材",
            "视觉风格流派",
            "媒介形式",
            "资产用途",
            "目标受众",
            "来源版权",
        ),
        title_must_exclude=(
            *_GENERIC_TITLE_TERMS,
            "海报",
            "插画",
            "动漫",
            "主题",
            "场景",
            "宣传",
            "构图组",
        ),
    ),
    "mood_atmosphere": ClusterSummaryDimensionPolicy(
        description_focus="情绪倾向、心理感受、氛围强度、节奏感和紧张或舒缓程度",
        title_focus="核心情绪或氛围",
        description_must_exclude=(
            "具体主体身份",
            "场景题材",
            "颜色构图",
            "媒介形式",
            "资产用途",
            "目标受众",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "配色", "构图", "海报", "插画", "主题"),
    ),
    "character_state_or_psychology": ClusterSummaryDimensionPolicy(
        description_focus="人物可观察的姿态、动作、表情、互动状态及有证据支持的心理线索",
        title_focus="人物的核心状态、动作或心理倾向",
        description_must_exclude=(
            "无证据的人物身份",
            "场景题材",
            "视觉风格",
            "颜色构图",
            "媒介形式",
            "资产用途",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "风格", "配色", "海报", "插画", "主题"),
    ),
    "asset_usage": ClusterSummaryDimensionPolicy(
        description_focus="资产的预期用途、使用场景、交付目标和工作流环节",
        title_focus="核心用途或交付目标",
        description_must_exclude=(
            "与用途无关的主体细节",
            "颜色构图",
            "情绪氛围",
            "无证据的受众",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "配色", "构图", "氛围"),
    ),
    "target_audience": ClusterSummaryDimensionPolicy(
        description_focus="有证据支持的目标人群、兴趣偏好、使用情境和传播对象",
        title_focus="核心目标受众",
        description_must_exclude=(
            "无证据的年龄、性别、地域和敏感属性",
            "具体主体内容",
            "颜色构图",
            "媒介形式",
            "来源版权",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "配色", "构图", "场景", "风格"),
    ),
    "provenance": ClusterSummaryDimensionPolicy(
        description_focus="有明确证据的素材来源、创作方式、派生关系、参考关系和采集路径",
        title_focus="核心来源或创作关系",
        description_must_exclude=(
            "未明确提供的作者身份",
            "未明确提供的版权结论",
            "主体内容",
            "颜色构图",
            "情绪氛围",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "配色", "构图", "场景", "风格"),
    ),
    "rights_version_authorship": ClusterSummaryDimensionPolicy(
        description_focus="有明确元数据支持的权利状态、授权范围、版本关系、作者和署名信息",
        title_focus="核心权利、版本或作者关系",
        description_must_exclude=(
            "无证据的权利或作者推断",
            "主体内容",
            "场景题材",
            "视觉风格",
            "颜色构图",
            "情绪氛围",
        ),
        title_must_exclude=(*_GENERIC_TITLE_TERMS, "配色", "构图", "场景", "风格"),
    ),
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
    policy = CLUSTER_SUMMARY_DIMENSION_POLICIES.get(
        embedding_type,
        ClusterSummaryDimensionPolicy(
            description_focus=f"仅限 {dimension_label} 这一语义维度中的共同特征",
            title_focus=f"仅限 {dimension_label} 维度中最有区分度的关键词",
            description_must_exclude=("其他语义维度的信息",),
        ),
    )
    payload = {
        "embedding_type": embedding_type,
        "semantic_dimension": dimension_label,
        "dimension_policy": {
            "description_focus": policy.description_focus,
            "title_focus": policy.title_focus,
            "description_must_exclude": list(policy.description_must_exclude),
            "title_must_exclude": list(policy.title_must_exclude),
        },
        "cluster_statistics": {
            "member_count": member_count,
            "average_membership_probability": round(average_membership_probability, 4),
        },
        "representative_assets": [
            _representative_evidence(asset, embedding_type=embedding_type)
            for asset in representatives
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "你正在总结一个通过单一 Feature 向量维度聚类发现的个人资产组。"
                "embedding_type 和 dimension_policy 是不可跨越的语义边界；所有输出字段只能"
                "使用 representative_assets 中当前维度的证据，不得混入主体、场景、风格、颜色、"
                "情绪、媒介、用途、受众、来源或版权等其他维度信息。严格按以下顺序完成："
                "第一步，先写 description，只概括 description_focus 指定的共同特征，并说明"
                "该维度内的必要差异；使用中文 50 到 150 字。第二步，再从已经写好的 description"
                "中提炼 name；name 不得增加 description 中没有的信息，只保留 title_focus 指定"
                "的关键词，简洁、可区分，不添加“图像、图片、素材、作品、集合、类别”等泛化尾词。"
                "keywords 和 common_features 也必须严格属于当前维度。description_must_exclude "
                "和 title_must_exclude 中列出的内容禁止出现在对应字段。只返回合法 JSON，并按"
                "“先描述、后标题”的字段顺序输出："
                '{"description":"...","name":"...","keywords":["..."],'
                '"common_features":["..."],"internal_variance":"low|medium|high"}。'
                "keywords 必须有 3 到 8 个。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _representative_evidence(
    asset: ClusterSummaryRepresentative,
    *,
    embedding_type: str,
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "asset_id": asset.asset_id,
        "role": asset.role.value,
        "membership_probability": round(asset.membership_probability, 4),
        "distance_to_medoid": round(asset.distance_to_medoid, 6),
    }
    if embedding_type == "native_multimodal":
        evidence["asset_name"] = asset.asset_name
        evidence["asset_description"] = asset.asset_description
    elif embedding_type == "asset_description":
        evidence["asset_description"] = asset.asset_description
    else:
        evidence["current_dimension_feature"] = _effective_feature(
            asset.asset_features.get(embedding_type)
        )
    return evidence


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
