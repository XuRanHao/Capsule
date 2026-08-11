"""Build complete text evidence used to name one Cluster Capsule."""

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from capsule.enums import EmbeddingType
from capsule.features import FEATURE_DIMENSION_SCOPES, effective_feature_text
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


CLUSTER_SUMMARY_DIMENSION_POLICIES: dict[str, ClusterSummaryDimensionPolicy] = {
    "native_multimodal": ClusterSummaryDimensionPolicy(
        description_focus="跨模态内容中共同出现的核心语义、对象关系、行为和上下文",
        title_focus="最能区分该簇的核心内容语义",
    ),
    "asset_description": ClusterSummaryDimensionPolicy(
        description_focus="资产自然语言描述中反复出现的事实、对象、行为和语义关系",
        title_focus="自然语言描述中的核心共同语义",
    ),
    "subject_content": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT],
        title_focus="最有区分度的主体、内容事实、主体动作或主体关系",
    ),
    "scene_theme": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME],
        title_focus="核心场景、事件或叙事情境",
    ),
    "visual_style": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_STYLE],
        title_focus="最有区分度的媒介形态、艺术技法、视觉语言或渲染质感",
    ),
    "color_composition": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.COLOR_COMPOSITION],
        title_focus="最有区分度的色彩关系、光线组织、视角、布局或视觉重心",
    ),
    "mood_atmosphere": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.MOOD_ATMOSPHERE],
        title_focus="整幅内容最稳定、最有区分度的情绪基调或感官氛围",
    ),
    "character_state_or_psychology": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY],
        title_focus="人物最有区分度的表情、身体状态、姿态、神态或心理状态",
    ),
    "asset_usage": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.ASSET_USAGE],
        title_focus="最具体的交付物、使用载体、制作任务、工作流环节或参考目的",
    ),
    "target_audience": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.TARGET_AUDIENCE],
        title_focus="证据最明确的观看者、使用者、兴趣群体或传播对象",
    ),
    "provenance": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.PROVENANCE],
        title_focus="证据最明确的来源平台、采集渠道、生成方式或派生关系",
    ),
    "rights_version_authorship": ClusterSummaryDimensionPolicy(
        description_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.RIGHTS_VERSION_AUTHORSHIP],
        title_focus="证据最明确的作者、所有权、授权、版权、版本或署名关系",
    ),
}


@dataclass(slots=True, frozen=True)
class ClusterSummaryAsset:
    """One cluster member's complete description and current-dimension evidence."""

    asset_id: str
    asset_description: str | None
    asset_features: dict[str, Any]
    source_relative_path: str = ""


def build_cluster_summary_messages(
    *,
    embedding_type: str,
    member_count: int,
    average_membership_probability: float,
    assets: Sequence[ClusterSummaryAsset],
    member_source_paths: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Create a text-only prompt containing every Asset in the cluster."""
    if not assets:
        raise ValueError("a Cluster Capsule summary requires cluster Assets")

    dimension_label = EMBEDDING_DIMENSION_LABELS.get(embedding_type, embedding_type)
    policy = CLUSTER_SUMMARY_DIMENSION_POLICIES.get(
        embedding_type,
        ClusterSummaryDimensionPolicy(
            description_focus=f"{dimension_label}这一语义维度中的共同特征",
            title_focus=f"{dimension_label}维度中最有区分度的语义",
        ),
    )
    payload = {
        "embedding_type": embedding_type,
        "semantic_dimension": dimension_label,
        "dimension_policy": {
            "description_focus": policy.description_focus,
            "title_focus": policy.title_focus,
        },
        "cluster_statistics": {
            "member_count": member_count,
            "average_membership_probability": round(average_membership_probability, 4),
        },
        "cluster_assets": [
            _cluster_asset_evidence(asset, embedding_type=embedding_type) for asset in assets
        ],
    }
    if embedding_type in _PATH_AWARE_EMBEDDING_TYPES:
        payload["member_source_context"] = cluster_source_context(member_source_paths)
    path_instruction = (
        f"当前维度为 {embedding_type}。member_source_context 来自该簇全部成员的真实相对"
        "路径。semantic_path_terms 中被多个成员支持、且与当前维度直接相关的代表性语义实体，"
        "可以为 name 或 common_features 提供一个补充事实。description 直接表达簇的共同语义；"
        "成员数量、完整路径、文件名和目录统计作为证据元数据保留。路径证据采用稳定、有实际"
        "含义的语义词。"
        if embedding_type in _PATH_AWARE_EMBEDDING_TYPES
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "你正在总结一个由单一 Feature 向量维度聚类得到的资产簇。你的任务不是为这个"
                "簇撰写完整介绍，而是尽可能准确、完整且简洁地提取簇内成员共同具备的当前维度"
                "特征。embedding_type 和 dimension_policy 共同定义当前任务的正向语义范围。"
                "cluster_assets 中全部成员的资产描述和当前维度描述共同构成证据。"
                "第一步生成 common_features：提取 description_focus 指定的当前维度共同特征；"
                "优先保留在多个簇内资产中重复出现或语义一致的特征。每项表达一个独立事实，"
                "表达结构服从当前维度；局部属性"
                "确实需要明确归属时才使用主体锚点。按证据支持度和区分度排列；相近、同义或"
                "包含关系的特征合并。在证据允许范围内尽可能完整提取；如果只能"
                "确认一个共同特征，就只输出一个。common_features 必须有 1 到 3 项。"
                "第二步生成 description：概括 common_features 已出现的内容，使用 30 到 80 "
                "个中文字符，以能够覆盖共同特征的最短自然表达为准，直接陈述当前维度的共同"
                "语义。成员差异通过 "
                "internal_variance 表达。"
                "第三步生成 name：从 common_features 和 description 中提炼最有区分度的 "
                "1 到 2 个共同特征，使用具体、紧凑的语义名称。"
                f"{path_instruction}"
                "只返回合法 JSON，不要输出 keywords，也不要输出其他字段："
                '{"description":"...","name":"...","common_features":["..."],'
                '"internal_variance":"low|medium|high"}。'
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _cluster_asset_evidence(
    asset: ClusterSummaryAsset,
    *,
    embedding_type: str,
) -> dict[str, object]:
    evidence: dict[str, object | None] = {
        "asset_id": asset.asset_id,
        "asset_description": asset.asset_description,
        "current_dimension_description": _current_dimension_description(
            asset,
            embedding_type=embedding_type,
        ),
    }
    if embedding_type in _PATH_AWARE_EMBEDDING_TYPES:
        source_relative_path = asset.source_relative_path
        if embedding_type == "asset_usage":
            raw_usage = asset.asset_features.get("asset_usage")
            if isinstance(raw_usage, dict):
                source_path = raw_usage.get("source_path")
                if isinstance(source_path, str) and source_path.strip():
                    source_relative_path = source_path.strip()
        normalized_path = _normalized_source_path(source_relative_path)
        if normalized_path is not None:
            evidence["source_relative_path"] = normalized_path
            evidence["source_file_name"] = PurePosixPath(normalized_path).name
    return evidence


def _current_dimension_description(
    asset: ClusterSummaryAsset,
    *,
    embedding_type: str,
) -> str | None:
    if embedding_type in {"native_multimodal", "asset_description"}:
        return asset.asset_description
    return effective_feature_text(asset.asset_features, embedding_type)


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
    """Preserve only a shared semantic path entity in the generated name."""
    if embedding_type not in _PATH_AWARE_EMBEDDING_TYPES:
        return summary
    context = cluster_source_context(source_paths)
    name = _ensure_cluster_path_name(summary.name, context)
    if name == summary.name:
        return summary
    return summary.model_copy(update={"name": name})


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
