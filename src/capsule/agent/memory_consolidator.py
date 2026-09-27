"""Structured-model nodes for the deterministic Agent memory worker flow."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from capsule.agent.memory_contracts import (
    ConversationSummary,
    MemoryAdjustmentBatch,
    MemoryCandidate,
    MemoryConsolidation,
    MemoryMutation,
)
from capsule.agent.memory_worker import MemoryConsolidator
from capsule.db.agent_memory import (
    AgentConversationRepository,
    ClaimedMemoryConsolidation,
    MemoryMatch,
)
from capsule.model_clients.doubao import DoubaoClient


class _SummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=8_000)
    topic: str | None = Field(default=None, max_length=128)


class _CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=3)


class MemoryModel(Protocol):
    async def summarize(
        self,
        *,
        claimed: ClaimedMemoryConsolidation,
    ) -> _SummaryDraft: ...

    async def extract_candidates(
        self,
        *,
        scope: Literal["workspace", "global"],
        claimed: ClaimedMemoryConsolidation,
        summary: ConversationSummary,
    ) -> list[MemoryCandidate]: ...

    async def resolve_candidate(
        self,
        *,
        candidate: MemoryCandidate,
        matches: list[MemoryMatch],
    ) -> MemoryAdjustmentBatch: ...


class DoubaoMemoryModel:
    """Constrained structured-output adapter for memory consolidation prompts."""

    def __init__(
        self,
        client: DoubaoClient,
        *,
        max_topic_chars: int,
        max_output_tokens: int = 1_500,
    ) -> None:
        self._client = client
        self._max_topic_chars = max_topic_chars
        self._max_output_tokens = max_output_tokens

    async def summarize(self, *, claimed: ClaimedMemoryConsolidation) -> _SummaryDraft:
        payload = {
            "previous_summary": claimed.thread.summary,
            "active_workspace_topics": claimed.active_topics,
            "messages": [_message_payload(item) for item in claimed.messages],
            "covered_through_sequence": claimed.event.through_sequence,
        }
        draft = await self._client.generate_structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责压缩一个 Agent 会话。仅输出 JSON。summary 要保留近期目标、"
                        "已确认决定、未完成事项和必要上下文；较旧且未被再次提及的信息优先省略。"
                        "输入按 sequence 排序且包含创建时间，越近期的信息权重越高。"
                        "topic 是简短主题，不能包含规则、偏好或具体记忆。若当前会话延续"
                        "active_workspace_topics 中的主题，必须原样复用对应 topic；仅在确实"
                        "进入新情景时生成新 topic。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            output_type=_SummaryDraft,
            schema_name="agent_conversation_summary",
            max_output_tokens=self._max_output_tokens,
        )
        topic = draft.topic
        if topic is not None:
            topic = topic[: self._max_topic_chars].strip() or None
        return _SummaryDraft(summary=draft.summary, topic=topic)

    async def extract_candidates(
        self,
        *,
        scope: Literal["workspace", "global"],
        claimed: ClaimedMemoryConsolidation,
        summary: ConversationSummary,
    ) -> list[MemoryCandidate]:
        payload = {
            "scope": scope,
            "summary": summary.model_dump(mode="json"),
            "active_workspace_topics": claimed.active_topics,
            "messages": [_message_payload(item) for item in claimed.messages],
        }
        scope_guidance = (
            "只提取当前工作区可跨会话复用的规则、约束、偏好、决定、事实或流程。"
            if scope == "workspace"
            else "只提取明确跨工作区适用的信息；没有明确证据时返回空数组。"
        )
        batch = await self._client.generate_structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责从会话中提取结构化长期记忆。仅输出 JSON，最多三条。"
                        f"{scope_guidance} 不要提取临时任务、猜测、工具报错、密钥或口令。"
                        "每条必须可追溯到输入消息，initial_confidence 在 0 到 1 之间。"
                        "decay_rate 由服务端按作用域设置，无需自行判断。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            output_type=_CandidateBatch,
            schema_name=f"agent_{scope}_memory_candidates",
            max_output_tokens=self._max_output_tokens,
        )
        return [item for item in batch.candidates if item.scope == scope]

    async def resolve_candidate(
        self,
        *,
        candidate: MemoryCandidate,
        matches: list[MemoryMatch],
    ) -> MemoryAdjustmentBatch:
        payload = {
            "candidate": candidate.model_dump(mode="json"),
            "existing_memories": [asdict(item) for item in matches],
        }
        return await self._client.generate_structured(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你负责逐条比较一条候选记忆与最多三条 RAG 返回的既有记忆。仅输出 JSON。"
                        "必须为 existing_memories 中每个 memory_id 输出一条 adjustments 项，"
                        "不能遗漏、"
                        "重复或编造 ID。语义一致或候选是更具体表述时用 merge；候选与既有记忆"
                        "冲突且应"
                        "删除旧记忆时用 deactivate；存在未证实冲突时用 lower_confidence 并给负 "
                        "confidence_delta；"
                        "与候选无关时用 ignore_rel；旧记忆仍然有效、候选不应新增时用 ignore_old。"
                        "不要输出 create。全部既有记忆为 ignore_rel 时系统才创建候选；merge 会创建"
                        "一条新的"
                        "合并记忆并停用被合并的旧记忆，deactivate 只停用冲突的旧记忆。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            output_type=MemoryAdjustmentBatch,
            schema_name="agent_memory_adjustments",
            max_output_tokens=500,
        )


class StructuredMemoryConsolidator(MemoryConsolidator):
    """Summary first, then parallel scope extraction and per-candidate RAG."""

    def __init__(
        self,
        *,
        model: MemoryModel,
        repository: AgentConversationRepository,
        max_mutations: int,
        workspace_decay_rate: float = 0.002,
        global_decay_rate: float = 0.0005,
    ) -> None:
        if (
            max_mutations < 1
            or workspace_decay_rate < 0
            or global_decay_rate < 0
            or global_decay_rate > workspace_decay_rate
        ):
            raise ValueError("memory consolidation limits are invalid")
        self._model = model
        self._repository = repository
        self._max_mutations = max_mutations
        self._workspace_decay_rate = workspace_decay_rate
        self._global_decay_rate = global_decay_rate

    async def summarize(
        self,
        claimed: ClaimedMemoryConsolidation,
    ) -> ConversationSummary:
        draft = await self._model.summarize(claimed=claimed)
        return ConversationSummary(
            summary=draft.summary,
            topic=draft.topic,
            covered_sequence=claimed.event.through_sequence,
        )

    async def consolidate(
        self,
        claimed: ClaimedMemoryConsolidation,
        *,
        summary: ConversationSummary,
    ) -> MemoryConsolidation:
        workspace_candidates, global_candidates = await asyncio.gather(
            self._model.extract_candidates(
                scope="workspace",
                claimed=claimed,
                summary=summary,
            ),
            self._model.extract_candidates(
                scope="global",
                claimed=claimed,
                summary=summary,
            ),
        )
        candidates = sorted(
            [
                *[
                    item.model_copy(update={"decay_rate": self._workspace_decay_rate})
                    for item in workspace_candidates
                ],
                *[
                    item.model_copy(update={"decay_rate": self._global_decay_rate})
                    for item in global_candidates
                ],
            ],
            key=lambda item: item.initial_confidence,
            reverse=True,
        )[: self._max_mutations]
        matches = await asyncio.gather(
            *[
                self._repository.retrieve_memory_matches(
                    user_id=claimed.event.user_id,
                    workspace_id=claimed.event.workspace_id,
                    candidate=candidate,
                    limit=3,
                )
                for candidate in candidates
            ]
        )
        resolutions = await asyncio.gather(
            *[
                self._model.resolve_candidate(candidate=candidate, matches=list(items))
                for candidate, items in zip(candidates, matches, strict=True)
            ]
        )
        mutations = [
            mutation
            for candidate, items, resolution in zip(candidates, matches, resolutions, strict=True)
            for mutation in _mutations_from_adjustments(candidate, list(items), resolution)
        ]
        return MemoryConsolidation(summary=summary, mutations=mutations)


def _mutations_from_adjustments(
    candidate: MemoryCandidate,
    matches: list[MemoryMatch],
    resolution: MemoryAdjustmentBatch,
) -> list[MemoryMutation]:
    """Validate per-memory decisions and reduce them to durable mutations."""

    allowed_ids = {item.memory_id for item in matches}
    adjustments = {item.memory_id: item for item in resolution.adjustments}
    if (
        len(adjustments) != len(resolution.adjustments)
        or set(adjustments) != allowed_ids
    ):
        # A malformed model batch must not partially mutate durable memories.
        return []

    merges = [item for item in adjustments.values() if item.action == "merge"]
    if len(merges) > 1:
        return []

    mutations: list[MemoryMutation] = []
    if merges:
        merge = merges[0]
        mutations.extend(
            [
                MemoryMutation(action="create", candidate=candidate),
                MemoryMutation(action="deactivate", target_memory_id=merge.memory_id),
            ]
        )

    deactivations = [
        item for item in adjustments.values() if item.action == "deactivate"
    ]
    for adjustment in deactivations:
        mutations.append(
            MemoryMutation(
                action="deactivate",
                target_memory_id=adjustment.memory_id,
            )
        )

    for adjustment in adjustments.values():
        if adjustment.action != "lower_confidence":
            continue
        mutations.append(
            MemoryMutation(
                action="lower_confidence",
                target_memory_id=adjustment.memory_id,
                confidence_delta=min(0.0, adjustment.confidence_delta),
            )
        )

    if not mutations and all(item.action == "ignore_rel" for item in adjustments.values()):
        return [MemoryMutation(action="create", candidate=candidate)]
    return mutations


def _message_payload(message: Any) -> dict[str, Any]:
    return {
        "message_id": message.message_id,
        "sequence": message.sequence,
        "role": message.role,
        "content": message.content,
        "name": message.name,
        "created_at": message.created_at.isoformat(),
    }
