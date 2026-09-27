"""Evaluate long-memory adjustment decisions without exercising memory retrieval."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass

from capsule.agent.memory_consolidator import (
    DoubaoMemoryModel,
    _mutations_from_adjustments,
)
from capsule.agent.memory_contracts import (
    MemoryAdjustment,
    MemoryAdjustmentAction,
    MemoryAdjustmentBatch,
    MemoryCandidate,
)
from capsule.config import get_settings
from capsule.db.agent_memory import MemoryMatch
from capsule.model_clients.doubao import DoubaoClient


@dataclass(frozen=True, slots=True)
class AdjustmentCase:
    case_id: str
    candidate: MemoryCandidate
    matches: list[MemoryMatch]
    expected_adjustments: dict[str, MemoryAdjustmentAction]


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


_DISTRACTORS = [
    ("language", "项目回复使用中文"),
    ("export", "交付文档采用 Markdown 格式"),
    ("retention", "导入前保留原始文件"),
    ("timezone", "项目时间统一使用香港时区"),
    ("video", "视频切段使用连续聚类"),
    ("review", "图谱修改需要用户确认"),
]


def _build_case(
    *,
    action: MemoryAdjustmentAction,
    index: int,
    key: str,
    candidate_text: str,
    retrieved_text: str,
    kind: str = "constraint",
) -> AdjustmentCase:
    """Inject one relevant or context-adjacent RAG item plus two distractors."""

    related_id = f"{action}_{index}_related"
    related = _memory(related_id, key, retrieved_text, kind=kind)
    first_key, first_text = _DISTRACTORS[(index * 2) % len(_DISTRACTORS)]
    second_key, second_text = _DISTRACTORS[(index * 2 + 1) % len(_DISTRACTORS)]
    matches = [
        _memory(f"{action}_{index}_d1", first_key, first_text),
        _memory(f"{action}_{index}_d2", second_key, second_text),
    ]
    matches.insert(index % 3, related)
    expected = {item.memory_id: "ignore_rel" for item in matches}
    if action not in {"ignore_rel"}:
        expected[related_id] = action
    return AdjustmentCase(
        case_id=f"{action}_{index}_{key}",
        candidate=_candidate(key, candidate_text, kind=kind),
        matches=matches,
        expected_adjustments=expected,
    )


def _cases() -> list[AdjustmentCase]:
    merges = [
        (
            "reply_language",
            "该项目回复和文档默认使用中文",
            "项目沟通与文档统一使用中文",
            "preference",
        ),
        (
            "export_format",
            "项目交付文档统一导出 Markdown",
            "交付物默认采用 Markdown 格式",
            "constraint",
        ),
        (
            "source_retention",
            "素材处理前必须保留原始文件",
            "导入和处理前保留原始素材",
            "constraint",
        ),
        (
            "timezone",
            "项目排期和记录使用香港时区",
            "项目时间统一使用香港时区",
            "constraint",
        ),
        (
            "graph_review",
            "图谱写操作需用户确认",
            "修改图谱前必须获得用户确认",
            "constraint",
        ),
        (
            "video_segmentation",
            "视频切段采用连续关键帧聚类",
            "视频分段使用连续帧聚类策略",
            "constraint",
        ),
    ]
    creates = [
        (
            "export_naming",
            "导出文件命名采用 YYYYMMDD_主题",
            "交付文档采用 Markdown 格式",
            "constraint",
        ),
        (
            "graph_scope",
            "图谱实体必须标注数据来源",
            "图谱只保存当前项目的实体关系",
            "constraint",
        ),
        (
            "video_sampling",
            "视频分析以每秒四帧采样",
            "视频切段使用连续关键帧聚类",
            "constraint",
        ),
        (
            "backup_period",
            "项目备份保留 90 天",
            "导入前保留原始文件",
            "constraint",
        ),
        (
            "document_title",
            "报告标题使用项目名加日期",
            "交付文档采用 Markdown 格式",
            "constraint",
        ),
        (
            "review_owner",
            "图谱写操作由项目负责人复核",
            "图谱修改需要用户确认",
            "constraint",
        ),
    ]
    ignore_old = [
        ("reply_language", "项目回复仍然保持中文", "项目回复默认使用中文", "preference"),
        (
            "export_format",
            "当前交付格式仍按既有约定执行",
            "交付物默认采用 Markdown 格式",
            "constraint",
        ),
        ("timezone", "现阶段继续沿用香港时区", "项目时间统一使用香港时区", "constraint"),
        ("review_owner", "当前仍由项目负责人复核", "图谱写操作由项目负责人复核", "constraint"),
        ("backup_period", "备份期限暂时维持现有设置", "项目备份保留 90 天", "constraint"),
        ("video_sampling", "当前视频采样规则保持不变", "视频分析以每秒四帧采样", "constraint"),
    ]
    deactivations = [
        (
            "legacy_template",
            "旧版导出模板已经停用，不再用于项目交付",
            "项目交付默认使用旧版导出模板",
        ),
        (
            "api_v1",
            "项目已废弃 API V1，后续请求不得继续使用",
            "所有外部接口请求使用 API V1",
        ),
        ("legacy_naming", "旧的随机文件命名规则已经取消", "导出文件使用随机字符串命名"),
        (
            "legacy_sampling",
            "每秒一帧的旧视频采样策略已经停用",
            "视频分析每秒采样一帧",
        ),
        ("legacy_graph", "旧图谱分类规则已废弃", "实体按旧版分类规则归档"),
        ("legacy_storage", "本地临时存储策略已经停用", "处理结果写入本地临时目录"),
    ]
    lowers = [
        (
            "reply_language",
            "可能需要把项目回复改成英文，尚未确认",
            "项目回复默认使用中文",
            "preference",
        ),
        (
            "export_template",
            "听说可能继续使用旧版导出模板，但尚无确认",
            "项目当前使用新版导出模板",
            "constraint",
        ),
        (
            "timezone",
            "可能改用东京时区，但该消息未被确认",
            "项目时间统一使用香港时区",
            "constraint",
        ),
        (
            "backup_period",
            "有人说备份或许只保留 7 天，未经确认",
            "项目备份保留 90 天",
            "constraint",
        ),
        (
            "review_owner",
            "可能不需要负责人复核，尚未确认",
            "图谱写操作由项目负责人复核",
            "constraint",
        ),
        (
            "video_sampling",
            "听说视频可能每秒只采样一帧，未经确认",
            "视频分析以每秒四帧采样",
            "constraint",
        ),
    ]
    return [
        *[
            _build_case(
                action="merge",
                index=index,
                key=key,
                candidate_text=candidate_text,
                retrieved_text=retrieved_text,
                kind=kind,
            )
            for index, (key, candidate_text, retrieved_text, kind) in enumerate(merges, start=1)
        ],
        *[
            _build_case(
                action="ignore_rel",
                index=index,
                key=key,
                candidate_text=candidate_text,
                retrieved_text=retrieved_text,
                kind=kind,
            )
            for index, (key, candidate_text, retrieved_text, kind) in enumerate(creates, start=1)
        ],
        *[
            _build_case(
                action="deactivate",
                index=index,
                key=key,
                candidate_text=candidate_text,
                retrieved_text=retrieved_text,
            )
            for index, (key, candidate_text, retrieved_text) in enumerate(deactivations, start=1)
        ],
        *[
            _build_case(
                action="ignore_old",
                index=index,
                key=key,
                candidate_text=candidate_text,
                retrieved_text=retrieved_text,
                kind=kind,
            )
            for index, (key, candidate_text, retrieved_text, kind) in enumerate(ignore_old, start=1)
        ],
        *[
            _build_case(
                action="lower_confidence",
                index=index,
                key=key,
                candidate_text=candidate_text,
                retrieved_text=retrieved_text,
                kind=kind,
            )
            for index, (key, candidate_text, retrieved_text, kind) in enumerate(lowers, start=1)
        ],
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


def _predicted_adjustments(
    case: AdjustmentCase,
    batch: MemoryAdjustmentBatch,
) -> tuple[dict[str, str], bool]:
    """Map model output to every injected RAG memory and flag malformed batches."""

    expected_ids = {item.memory_id for item in case.matches}
    by_id: dict[str, str] = {}
    malformed = False
    for adjustment in batch.adjustments:
        if adjustment.memory_id not in expected_ids or adjustment.memory_id in by_id:
            malformed = True
            continue
        by_id[adjustment.memory_id] = adjustment.action
    for memory_id in expected_ids:
        if memory_id not in by_id:
            by_id[memory_id] = "invalid"
            malformed = True
    return by_id, malformed


def _mutation_signature(case: AdjustmentCase, batch: MemoryAdjustmentBatch) -> list[str]:
    mutations = _mutations_from_adjustments(case.candidate, case.matches, batch)
    return [
        mutation.action
        if mutation.target_memory_id is None
        else f"{mutation.action}:{mutation.target_memory_id}"
        for mutation in mutations
    ]


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

    expected_actions: list[str] = []
    predicted_actions: list[str] = []
    results: list[dict[str, object]] = []
    strict_case_correct: list[bool] = []
    for case, resolution in zip(cases, resolutions, strict=True):
        predicted, malformed = _predicted_adjustments(case, resolution)
        ordered_expected = [case.expected_adjustments[item.memory_id] for item in case.matches]
        ordered_predicted = [predicted[item.memory_id] for item in case.matches]
        expected_actions.extend(ordered_expected)
        predicted_actions.extend(ordered_predicted)
        strict = not malformed and ordered_expected == ordered_predicted
        strict_case_correct.append(strict)
        results.append(
            {
                "case_id": case.case_id,
                "candidate": case.candidate.display_text,
                "expected_adjustments": [
                    {
                        "memory_id": item.memory_id,
                        "action": case.expected_adjustments[item.memory_id],
                    }
                    for item in case.matches
                ],
                "predicted_adjustments": [
                    {"memory_id": item.memory_id, "action": predicted[item.memory_id]}
                    for item in case.matches
                ],
                "expected_mutations": _mutation_signature(
                    case,
                    MemoryAdjustmentBatch(
                        adjustments=[
                            MemoryAdjustment(memory_id=memory_id, action=action)
                            for memory_id, action in case.expected_adjustments.items()
                        ]
                    ),
                ),
                "predicted_mutations": _mutation_signature(case, resolution),
                "strict_case_correct": strict,
            }
        )
    report = {
        "evaluation": "memory_adjustment_only",
        "candidate_cases": len(cases),
        "rag_memory_pairs": len(expected_actions),
        "pair_action_macro_f1": round(_macro_f1(expected_actions, predicted_actions), 4),
        "strict_case_accuracy": round(sum(strict_case_correct) / len(cases), 4),
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
