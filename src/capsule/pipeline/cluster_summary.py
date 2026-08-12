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

EMBEDDING_DIMENSION_LABELS: dict[str, str] = {
    "native_multimodal": "跨模态内容语义",
    "asset_description": "资产自然语言描述",
    "subject_content": "主体与内容",
    "scene_theme": "场景与题材",
    "visual_presentation": "视觉表现",
    "visual_style": "视觉风格",
    "color_composition": "色彩与构图",
    "mood_atmosphere": "画面情绪氛围",
    "character_state_or_psychology": "人物状态或心理",
    "asset_usage": "资产用途",
    "target_audience": "目标受众",
    "provenance": "来源与创作关系",
    "rights_version_authorship": "权利、版本与作者",
}

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
    """The shared semantic boundary for one embedding channel's summary."""

    summary_focus: str


CLUSTER_SUMMARY_DIMENSION_POLICIES: dict[str, ClusterSummaryDimensionPolicy] = {
    "native_multimodal": ClusterSummaryDimensionPolicy(
        summary_focus="跨模态内容中共同出现且最能区分该簇的核心语义、对象关系、行为和上下文",
    ),
    "subject_content": ClusterSummaryDimensionPolicy(
        summary_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.SUBJECT_CONTENT],
    ),
    "scene_theme": ClusterSummaryDimensionPolicy(
        summary_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.SCENE_THEME],
    ),
    "visual_presentation": ClusterSummaryDimensionPolicy(
        summary_focus=FEATURE_DIMENSION_SCOPES[EmbeddingType.VISUAL_PRESENTATION],
    ),
}


@dataclass(slots=True, frozen=True)
class ClusterSummaryAsset:
    """One cluster member's complete description and current-dimension evidence."""

    asset_id: str
    asset_description: str | None
    asset_features: dict[str, Any]
    source_relative_path: str = ""
    file_tree_context: tuple[str, ...] = ()


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
            summary_focus=f"{dimension_label}这一维度中共同且有区分度的语义",
        ),
    )
    payload = {
        "embedding_type": embedding_type,
        "semantic_dimension": dimension_label,
        "dimension_policy": {
            "summary_focus": policy.summary_focus,
        },
        "cluster_statistics": {
            "member_count": member_count,
            "average_membership_probability": round(average_membership_probability, 4),
        },
        "cluster_assets": [
            _cluster_asset_evidence(asset, embedding_type=embedding_type) for asset in assets
        ],
    }
    payload["member_metadata_context"] = cluster_source_context(member_source_paths)
    return [
        {
            "role": "system",
            "content": (
                "你正在为一个由单一 Feature 向量维度聚类得到的资产簇生成簇名和描述。"
                "cluster_assets 包含全部成员的内容证据和文件元数据；member_metadata_context "
                "汇总了簇内真实相对路径、目录和文件名。先判断这些元数据能否可靠体现用户组织"
                "素材的意图，例如项目、作品、角色、系列、主题或资产类别。有效元数据必须具备"
                "实际语义，并由多个成员重复支持，或能与多个成员的内容证据互相印证。文件扩展"
                "名、编号、日期、UUID、通用目录、临时/导出标记和孤立的单个命名都不是有效意图"
                "证据；不得机械截取共同字符串，也不得臆造专名或归属。"
                "如果能够提取可信的用户意图，就以该意图为核心锚点，同时生成彼此一致的 name "
                "和 description：name 优先使用用户元数据中的具体称谓，并结合簇内资产范围形成"
                "紧凑自然的名称；description 说明该意图下本簇汇集了哪些共同内容，以及当前"
                "embedding_type 所体现的组织范围。比如多个文件名分别表达“风不觉-武器”、"
                "“风不觉-角色设定”和“风不觉-时装”时，应理解为用户在组织“风不觉”相关资产，"
                "簇名与描述都应围绕“风不觉”生成，而不是只取某个视觉共同点。"
                "如果根据现有元数据无法提取可信意图，则完全回退到原有按聚类维度总结的路径："
                "embedding_type 和 dimension_policy 共同定义正向语义范围，从全部成员的资产"
                "描述和 current_dimension_description 中提取共同特征，并据此同时生成 name 和 "
                "description。name 使用最有区分度的 1 到 2 个共同特征；description 使用 30 到 "
                "80 个中文字符，以覆盖共同语义的最短自然表达为准。"
                "无论采用哪条路径，name 和 description 都必须作为同一次判断的整体直接生成，"
                "不能先生成其中一个再用另一个二次改写。common_features 记录支撑该命名和描述"
                "的 1 到 3 项事实，按证据支持度和区分度排列并合并同义项；成员差异通过 "
                "internal_variance 表达。"
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
    normalized_path = _normalized_source_path(asset.source_relative_path)
    if normalized_path is not None:
        evidence["source_relative_path"] = normalized_path
        evidence["source_file_name"] = PurePosixPath(normalized_path).name
    file_tree_context = [part.strip() for part in asset.file_tree_context if part.strip()]
    if file_tree_context:
        evidence["file_tree_context"] = file_tree_context
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
