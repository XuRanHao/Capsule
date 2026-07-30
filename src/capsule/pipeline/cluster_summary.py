"""Build the deliberately small model input used to name one Cluster Capsule."""

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from capsule.enums import ClusterRepresentativeRole
from capsule.features import effective_feature_text
from capsule.schemas import ClusterSummary

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

_PATH_AWARE_EMBEDDING_TYPES = frozenset({"subject_content", "asset_usage"})
_GENERIC_PATH_TERMS = frozenset(
    {
        "asset",
        "assets",
        "image",
        "images",
        "img",
        "picture",
        "pictures",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "gif",
        "video",
        "mp4",
        "mov",
        "reference",
        "references",
        "ref",
        "source",
        "sources",
        "temp",
        "tmp",
        "export",
        "exports",
        "素材",
        "图片",
        "图像",
        "文件",
        "参考",
        "海报",
        "立绘",
        "设定",
        "角色",
        "人物",
        "成品",
        "草稿",
        "正稿",
        "导出",
        "编辑",
        "小说编辑",
        "测试",
        "戴帽子",
        "无帽子",
    }
)
_GENERIC_PATH_SUFFIXES = (
    "参考",
    "素材",
    "图片",
    "图像",
    "文件",
    "导出",
    "编辑",
    "内部",
    "外部",
    "大厅",
    "场景",
    "镜头",
    "海报",
    "立绘",
    "设定",
    "三视图",
    "合集",
    "作品",
    "成品",
    "草稿",
    "正稿",
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
        description_focus=(
            "主体对象、人物或物体特征、动作、主体关系，以及成员相对路径中明确标注且"
            "与主体证据一致的项目名、角色名和代表文件信息"
        ),
        title_focus="路径明确标注的项目或角色实体名，加上核心主体或主体动作",
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
        description_focus=(
            "资产的预期用途、使用场景、交付目标、工作流环节，以及成员素材共同的"
            "相对目录路径、路径中明确标注的项目或角色实体名和代表文件信息"
        ),
        title_focus="路径明确标注的项目或角色实体名，加上核心用途或交付目标",
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
    source_relative_path: str = ""


def build_cluster_summary_messages(
    *,
    embedding_type: str,
    member_count: int,
    average_membership_probability: float,
    representatives: list[ClusterSummaryRepresentative],
    member_source_paths: Sequence[str] = (),
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
    if embedding_type in _PATH_AWARE_EMBEDDING_TYPES:
        payload["member_source_context"] = cluster_source_context(member_source_paths)
    description_length_instruction = (
        "使用中文 50 到 90 字"
        if embedding_type in _PATH_AWARE_EMBEDDING_TYPES
        else "使用中文 50 到 150 字"
    )
    path_instruction = (
        f"当前维度为 {embedding_type}。member_source_context 来自该簇全部成员的真实相对"
        "路径。description 必须明确写出 semantic_path_terms 中与当前维度证据一致的项目名"
        "或角色名及对应成员数量，但不要在模型 description 中复述完整 relative_path 或"
        "file_name；系统会在保存前确定性追加代表文件与相对路径，避免超过长度上限。name "
        "必须优先保留上述有语义的项目名或角色名，再概括当前维度，例如“古小玲少女立绘”"
        "或“古小玲角色立绘制作”；如果只有部分成员属于该实体，使用“古小玲及其他……”等"
        "不误导的表达。哈希、纯数字、日期文件名不得进入 name。路径只能作为当前维度事实"
        "的补充证据，不得据此混入其他维度；directory_counts 中未进入 semantic_path_terms "
        "的“参考、测试、png、小说编辑”等通用目录词不得进入 name，完整路径中的上层目录"
        "也不得被误判为当前资产的角色身份。"
        if embedding_type in _PATH_AWARE_EMBEDDING_TYPES
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "你正在总结一个通过单一 Feature 向量维度聚类发现的个人资产组。"
                "embedding_type 和 dimension_policy 是不可跨越的语义边界；所有输出字段只能"
                "使用 representative_assets 中当前维度的证据，不得混入主体、场景、风格、颜色、"
                "情绪、媒介、用途、受众、来源或版权等其他维度信息。严格按以下顺序完成："
                "第一步，先写 description，只概括 description_focus 指定的共同特征，并说明"
                f"该维度内的必要差异；{description_length_instruction}。第二步，再从已经写好的 "
                "description 中提炼 name；name 不得增加 description 中没有的信息，只保留 "
                "title_focus 指定的关键词，简洁、可区分，不添加“图像、图片、素材、作品、"
                "集合、类别”等泛化尾词。"
                "name 不得堆叠近义词、同义词或只有措辞差异的词组；同一概念只保留一个最准确"
                f"的短语。{path_instruction}"
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
        evidence["current_dimension_feature"] = effective_feature_text(
            asset.asset_features,
            embedding_type,
        )
        if embedding_type in _PATH_AWARE_EMBEDDING_TYPES:
            source_relative_path = asset.source_relative_path
            if embedding_type == "asset_usage":
                raw_usage = asset.asset_features.get("asset_usage")
                if isinstance(raw_usage, dict):
                    description = raw_usage.get("description")
                    if isinstance(description, str) and description.strip():
                        evidence["asset_usage_description"] = description.strip()
                    source_path = raw_usage.get("source_path")
                    if isinstance(source_path, str) and source_path.strip():
                        source_relative_path = source_path.strip()
            normalized_path = _normalized_source_path(source_relative_path)
            if normalized_path is not None:
                evidence["source_relative_path"] = normalized_path
                evidence["source_file_name"] = PurePosixPath(normalized_path).name
    return evidence


def cluster_source_context(source_paths: Sequence[str]) -> dict[str, object]:
    """Summarize relative paths, meaningful directory labels, and file names."""
    normalized_paths = [
        path for raw_path in source_paths if (path := _normalized_source_path(raw_path)) is not None
    ]
    unique_paths = list(dict.fromkeys(normalized_paths))
    directory_counts = Counter(
        directory for path in normalized_paths if (directory := _source_directory(path))
    )
    semantic_term_counts: Counter[str] = Counter()
    for path in normalized_paths:
        terms = _semantic_path_terms(PurePosixPath(path).parent.parts)
        if terms:
            semantic_term_counts[terms[-1]] += 1
    ranked_terms = sorted(
        semantic_term_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    representative_paths = sorted(
        unique_paths,
        key=lambda path: (
            not _is_human_readable_file_name(PurePosixPath(path).name),
            path,
        ),
    )
    return {
        "member_count_with_path": len(normalized_paths),
        "directory_counts": [
            {"directory": directory, "member_count": count}
            for directory, count in directory_counts.most_common(5)
        ],
        "semantic_path_terms": [
            {"term": term, "member_count": count} for term, count in ranked_terms[:8] if count >= 2
        ],
        "representative_files": [
            {
                "file_name": PurePosixPath(path).name,
                "relative_path": path,
            }
            for path in representative_paths[:5]
        ],
    }


def asset_usage_path_context(source_paths: Sequence[str]) -> dict[str, object]:
    """Backward-compatible alias for the shared path context."""
    return cluster_source_context(source_paths)


def ensure_path_aware_cluster_summary(
    summary: ClusterSummary,
    source_paths: Sequence[str],
    *,
    embedding_type: str,
) -> ClusterSummary:
    """Guarantee path-aware dimensions persist source identity and file evidence."""
    if embedding_type not in _PATH_AWARE_EMBEDDING_TYPES:
        return summary
    context = cluster_source_context(source_paths)
    description = _ensure_cluster_path_description(summary.description, context)
    name = _ensure_cluster_path_name(summary.name, context)
    if name == summary.name and description == summary.description:
        return summary
    return summary.model_copy(update={"name": name, "description": description})


def ensure_asset_usage_cluster_path_description(
    description: str,
    source_paths: Sequence[str],
    *,
    max_length: int = 150,
) -> str:
    """Guarantee the persisted usage Capsule visibly names its path evidence."""
    return _ensure_cluster_path_description(
        description,
        cluster_source_context(source_paths),
        max_length=max_length,
    )


def _ensure_cluster_path_description(
    description: str,
    context: dict[str, object],
    *,
    max_length: int = 150,
) -> str:
    raw_directories = context["directory_counts"]
    directories = (
        [item for item in raw_directories if isinstance(item, dict)]
        if isinstance(raw_directories, list)
        else []
    )
    raw_terms = context["semantic_path_terms"]
    semantic_terms = (
        [item for item in raw_terms if isinstance(item, dict)]
        if isinstance(raw_terms, list)
        else []
    )
    raw_files = context["representative_files"]
    representative_files = (
        [item for item in raw_files if isinstance(item, dict)]
        if isinstance(raw_files, list)
        else []
    )
    if not representative_files:
        return description

    primary_term = str(semantic_terms[0].get("term", "")) if semantic_terms else ""
    primary_count = int(semantic_terms[0].get("member_count", 0)) if semantic_terms else 0
    primary_directory = str(directories[0].get("directory", "")) if directories else ""
    representative = _representative_file_for_context(
        representative_files,
        term=primary_term,
        directory=primary_directory,
    )
    representative_path = str(representative.get("relative_path", ""))
    file_name = str(representative.get("file_name", ""))
    if representative_path in description and file_name in description:
        return description
    directory = _source_directory(representative_path)
    if primary_term:
        prefix = (
            f"路径实体「{primary_term}」出现在{primary_count}项成员中；代表文件"
            f"「{file_name}」位于相对路径「{representative_path}」。"
        )
    elif directories:
        directory_count = int(directories[0].get("member_count", 0))
        prefix = (
            f"成员中{directory_count}项来自「{directory}」；代表文件「{file_name}」"
            f"位于相对路径「{representative_path}」。"
        )
    else:
        prefix = f"代表文件「{file_name}」位于相对路径「{representative_path}」。"
    prefix_budget = max(40, max_length - 50)
    if len(prefix) > prefix_budget:
        identity = f"路径实体「{primary_term}」覆盖{primary_count}项；" if primary_term else ""
        prefix = f"{identity}相对目录「{directory}」，代表文件「{file_name}」。"
    if len(prefix) > prefix_budget:
        file_budget = max(12, prefix_budget - len(directory) - 25)
        prefix = (
            f"相对目录「{_abbreviate(directory, 30)}」，"
            f"代表文件「{_abbreviate(file_name, file_budget)}」。"
        )
    remaining = max_length - len(prefix)
    if remaining <= 0:
        return _abbreviate(prefix, max_length)
    return f"{prefix}{_truncate_complete_sentence(description, remaining)}".rstrip()


def _ensure_cluster_path_name(name: str, context: dict[str, object]) -> str:
    raw_terms = context["semantic_path_terms"]
    semantic_terms = (
        [item for item in raw_terms if isinstance(item, dict)]
        if isinstance(raw_terms, list)
        else []
    )
    if not semantic_terms:
        return name
    primary_term = str(semantic_terms[0].get("term", "")).strip()
    if not primary_term or primary_term in name:
        return name
    primary_count = int(semantic_terms[0].get("member_count", 0))
    raw_member_count = context.get("member_count_with_path", 0)
    member_count = raw_member_count if isinstance(raw_member_count, int) else 0
    if primary_count < member_count:
        return f"{primary_term}及其他{name}"
    return f"{primary_term}{name}"


def _representative_file_for_context(
    representative_files: list[dict[str, object]],
    *,
    term: str,
    directory: str,
) -> dict[str, object]:
    if term:
        for representative in representative_files:
            path = str(representative.get("relative_path", ""))
            if term in PurePosixPath(path).parent.parts:
                return representative
    if directory:
        for representative in representative_files:
            path = str(representative.get("relative_path", ""))
            if _source_directory(path) == directory:
                return representative
    return representative_files[0]


def _semantic_path_terms(parts: Sequence[str]) -> list[str]:
    return [part.strip() for part in parts if _is_semantic_path_term(part)]


def _is_semantic_path_term(raw_term: str) -> bool:
    term = raw_term.strip()
    lowered = term.casefold()
    if not term or lowered in _GENERIC_PATH_TERMS:
        return False
    if re.fullmatch(r"第?[0-9一二三四五六七八九十百]+[集章节季期幕版]", term):
        return False
    if re.fullmatch(r"[0-9a-fA-F]{8,}", term):
        return False
    if any(character.isdigit() for character in term):
        return False
    if any(term.endswith(suffix) for suffix in _GENERIC_PATH_SUFFIXES):
        return False
    if re.search(r"[\u3400-\u9fff]", term):
        return 2 <= len(term) <= 12
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z _-]{2,30}", term))


def _is_human_readable_file_name(file_name: str) -> bool:
    stem = PurePosixPath(file_name).stem.strip()
    if not stem or re.fullmatch(r"[0-9a-fA-F]{8,}", stem):
        return False
    if re.fullmatch(r"[\d_\-. √]+", stem):
        return False
    return bool(re.search(r"[\u3400-\u9fffA-Za-z]", stem))


def _abbreviate(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 1:
        return "…"
    head_length = (max_length - 1) // 2
    tail_length = max_length - head_length - 1
    return f"{value[:head_length]}…{value[-tail_length:]}"


def _truncate_complete_sentence(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        if value.endswith(("。", "！", "？", "!", "?")):
            return value
        if len(value) < max_length:
            return f"{value.rstrip('，、；：,;: ')}。"
        return f"{value[: max_length - 1].rstrip('，、；：,;: ')}。"
    if max_length <= 1:
        return "。"
    candidate = value[: max_length - 1].rstrip("，、；：,;: ")
    boundaries = {
        mark: candidate.rfind(mark) for mark in ("。", "！", "？", "!", "?", "；", ";", "，", ",")
    }
    boundary_mark, last_boundary = max(boundaries.items(), key=lambda item: item[1])
    if last_boundary >= max_length // 2:
        complete_clause = candidate[: last_boundary + 1]
        if boundary_mark in {"；", ";", "，", ","}:
            complete_clause = f"{complete_clause[:-1]}。"
        return complete_clause
    return f"{candidate}。"


def _normalized_source_path(raw_path: str) -> str | None:
    normalized = raw_path.strip().replace("\\", "/")
    if not normalized:
        return None
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _source_directory(source_path: str) -> str:
    directory = PurePosixPath(source_path).parent.as_posix()
    return "" if directory == "." else directory
