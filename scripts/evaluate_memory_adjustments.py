"""Evaluate long-memory adjustment decisions without exercising memory retrieval."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from capsule.agent.memory_consolidator import DoubaoMemoryModel
from capsule.agent.memory_contracts import MemoryCandidate
from capsule.config import get_settings
from capsule.db.agent_memory import MemoryMatch
from capsule.model_clients.doubao import DoubaoClient

ExpectedAction = Literal["create", "merge", "deactivate", "lower_confidence"]


@dataclass(frozen=True, slots=True)
class AdjustmentCase:
    case_id: str
    candidate: MemoryCandidate
    matches: list[MemoryMatch]
    expected_action: ExpectedAction
    expected_target_memory_id: str | None = None


def _candidate(key: str, text: str, *, kind: str = "constraint") -> MemoryCandidate:
    return MemoryCandidate(
        scope="workspace",
        kind=kind,
        memory_key=key,
        value={"statement": text},
        display_text=text,
        topics=["项目规范"],
        initial_confidence=0.9,
    )


def _memory(memory_id: str, key: str, text: str, *, kind: str = "constraint") -> MemoryMatch:
    return MemoryMatch(
        memory_id=memory_id,
        scope="workspace",
        kind=kind,
        memory_key=key,
        value={"statement": text},
        display_text=text,
        topics=["项目规范"],
        confidence=0.9,
        effective_confidence=0.9,
        decay_rate=0.002,
        status="active",
        # All candidates use the same supplied relevance: this benchmark does
        # not measure retrieval ordering or vector similarity.
        relevance=0.5,
    )


def _cases() -> list[AdjustmentCase]:
    return [
        AdjustmentCase(
            "merge_language",
            _candidate("reply_language", "该项目的回复和文档默认使用中文", kind="preference"),
            [
                _memory("m_unrelated_1", "export", "交付文档采用 Markdown 格式"),
                _memory(
                    "m_language",
                    "reply_language",
                    "项目沟通与文档统一使用中文",
                    kind="preference",
                ),
                _memory("m_unrelated_2", "retention", "导入前保留原始文件"),
            ],
            "merge",
            "m_language",
        ),
        AdjustmentCase(
            "merge_export",
            _candidate("export_format", "项目交付文档统一导出为 Markdown"),
            [
                _memory("m_unrelated_3", "language", "项目回复使用中文"),
                _memory("m_unrelated_4", "review", "图谱修改需要用户确认"),
                _memory("m_export", "export_format", "交付物默认使用 Markdown 格式"),
            ],
            "merge",
            "m_export",
        ),
        AdjustmentCase(
            "merge_source_retention",
            _candidate("source_retention", "素材处理前必须保留原始文件"),
            [
                _memory("m_retention", "source_retention", "导入和处理前保留原始素材"),
                _memory("m_unrelated_5", "timezone", "项目时间统一使用香港时区"),
                _memory("m_unrelated_6", "video", "视频切段使用连续聚类"),
            ],
            "merge",
            "m_retention",
        ),
        AdjustmentCase(
            "create_file_naming",
            _candidate("export_naming", "导出文件命名采用 YYYYMMDD_主题"),
            [
                _memory("m_unrelated_7", "language", "项目回复使用中文"),
                _memory("m_related_but_distinct_1", "export_format", "交付文档采用 Markdown 格式"),
                _memory("m_unrelated_8", "retention", "导入前保留原始文件"),
            ],
            "create",
        ),
        AdjustmentCase(
            "create_graph_review",
            _candidate("graph_review", "图谱写操作必须先获得用户确认"),
            [
                _memory("m_related_but_distinct_2", "graph_scope", "图谱只保存当前项目的实体关系"),
                _memory("m_unrelated_9", "language", "项目回复使用中文"),
                _memory("m_unrelated_10", "export", "交付文档采用 Markdown 格式"),
            ],
            "create",
        ),
        AdjustmentCase(
            "create_video_strategy",
            _candidate("video_segmentation", "视频切段采用连续关键帧聚类"),
            [
                _memory("m_unrelated_11", "retention", "导入前保留原始文件"),
                _memory("m_related_but_distinct_3", "video_sampling", "视频分析每秒采样两个关键帧"),
                _memory("m_unrelated_12", "language", "项目回复使用中文"),
            ],
            "create",
        ),
        AdjustmentCase(
            "deactivate_old_template",
            _candidate("legacy_template", "旧版导出模板已经停用，不再用于项目交付"),
            [
                _memory("m_unrelated_13", "language", "项目回复使用中文"),
                _memory("m_old_template", "legacy_template", "项目交付默认使用旧版导出模板"),
                _memory("m_unrelated_14", "retention", "导入前保留原始文件"),
            ],
            "deactivate",
            "m_old_template",
        ),
        AdjustmentCase(
            "deactivate_api_v1",
            _candidate("api_v1", "项目已废弃 API V1，后续请求不得继续使用"),
            [
                _memory("m_old_api", "api_v1", "所有外部接口请求使用 API V1"),
                _memory("m_unrelated_15", "export", "交付文档采用 Markdown 格式"),
                _memory("m_unrelated_16", "timezone", "项目时间统一使用香港时区"),
            ],
            "deactivate",
            "m_old_api",
        ),
        AdjustmentCase(
            "deactivate_legacy_naming",
            _candidate("legacy_naming", "旧的随机文件命名规则已经取消"),
            [
                _memory("m_unrelated_17", "language", "项目回复使用中文"),
                _memory("m_unrelated_18", "video", "视频切段使用连续聚类"),
                _memory("m_old_naming", "legacy_naming", "导出文件使用随机字符串命名"),
            ],
            "deactivate",
            "m_old_naming",
        ),
        AdjustmentCase(
            "lower_unverified_language",
            _candidate("reply_language", "可能需要把项目回复改成英文，尚未确认", kind="preference"),
            [
                _memory(
                    "m_confirmed_language",
                    "reply_language",
                    "项目回复默认使用中文",
                    kind="preference",
                ),
                _memory("m_unrelated_19", "export", "交付文档采用 Markdown 格式"),
                _memory("m_unrelated_20", "retention", "导入前保留原始文件"),
            ],
            "lower_confidence",
            "m_confirmed_language",
        ),
        AdjustmentCase(
            "lower_unverified_template",
            _candidate("export_template", "听说可能要继续使用旧版导出模板，但尚无确认"),
            [
                _memory("m_unrelated_21", "language", "项目回复使用中文"),
                _memory("m_current_template", "export_template", "项目当前使用新版导出模板"),
                _memory("m_unrelated_22", "video", "视频切段使用连续聚类"),
            ],
            "lower_confidence",
            "m_current_template",
        ),
        AdjustmentCase(
            "lower_unverified_timezone",
            _candidate("timezone", "可能改用东京时区，但该消息未被确认"),
            [
                _memory("m_unrelated_23", "export", "交付文档采用 Markdown 格式"),
                _memory("m_unrelated_24", "language", "项目回复使用中文"),
                _memory("m_confirmed_timezone", "timezone", "项目时间统一使用香港时区"),
            ],
            "lower_confidence",
            "m_confirmed_timezone",
        ),
    ]


def _macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores) if scores else 0.0


async def main() -> None:
    settings = get_settings()
    if settings.ark_api_key is None:
        raise RuntimeError("CAPSULE_ARK_API_KEY is required for this model evaluation")
    cases = _cases()
    async with DoubaoClient(settings) as client:
        model = DoubaoMemoryModel(
            client,
            max_topic_chars=settings.agent_memory_max_topic_chars,
        )
        resolutions = await asyncio.gather(
            *[
                model.resolve_candidate(candidate=case.candidate, matches=case.matches)
                for case in cases
            ]
        )

    expected_actions = [case.expected_action for case in cases]
    predicted_actions = [resolution.action for resolution in resolutions]
    strict_correct = [
        resolution.action == case.expected_action
        and resolution.target_memory_id == case.expected_target_memory_id
        for case, resolution in zip(cases, resolutions, strict=True)
    ]
    report = {
        "evaluation": "memory_adjustment_only",
        "cases": len(cases),
        "operation_macro_f1": round(_macro_f1(expected_actions, predicted_actions), 4),
        "strict_action_target_accuracy": round(sum(strict_correct) / len(cases), 4),
        "results": [
            {
                "case_id": case.case_id,
                "expected_action": case.expected_action,
                "expected_target_memory_id": case.expected_target_memory_id,
                "predicted_action": resolution.action,
                "predicted_target_memory_id": resolution.target_memory_id,
                "strict_correct": correct,
            }
            for case, resolution, correct in zip(cases, resolutions, strict_correct, strict=True)
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
