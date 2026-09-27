import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from capsule.agent.memory_consolidator import (
    DoubaoMemoryModel,
    StructuredMemoryConsolidator,
    _mutations_from_adjustments,
)
from capsule.agent.memory_contracts import (
    MemoryAdjustment,
    MemoryAdjustmentBatch,
    MemoryCandidate,
)
from capsule.db.agent_memory import (
    AgentMessageRecord,
    AgentThreadRecord,
    ClaimedMemoryConsolidation,
    MemoryMatch,
    MemoryOutboxEvent,
)


@dataclass
class _SummaryDraft:
    summary: str
    topic: str | None


def _candidate(scope: str, key: str, confidence: float) -> MemoryCandidate:
    return MemoryCandidate(
        scope=scope,  # type: ignore[arg-type]
        kind="policy",
        memory_key=key,
        value={"rule": key},
        display_text=f"记忆 {key}",
        topics=["主题"],
        initial_confidence=confidence,
        decay_rate=0.01,
    )


def _claimed() -> ClaimedMemoryConsolidation:
    return ClaimedMemoryConsolidation(
        event=MemoryOutboxEvent(
            event_id="event-1",
            thread_id="thread-1",
            user_id="user-1",
            workspace_id="workspace-1",
            through_sequence=2,
        ),
        messages=[
            AgentMessageRecord(
                message_id="message-1",
                thread_id="thread-1",
                sequence=1,
                role="user",
                content="请使用中文",
                name=None,
                turn_id="turn-1",
                request_id="request-1",
                estimated_tokens=4,
                created_at=datetime.now(UTC),
            )
        ],
        thread=AgentThreadRecord(
            thread_id="thread-1",
            user_id="user-1",
            workspace_id="workspace-1",
            title="新会话",
            status="active",
            summary=None,
            summary_topic=None,
            summary_covered_sequence=0,
            last_consolidated_sequence=0,
            memory_revision=0,
            last_message_at=None,
        ),
        active_topics=[],
    )


class _Model:
    def __init__(self) -> None:
        self.extract_scopes: list[str] = []
        self.resolved_keys: list[str] = []

    async def summarize(self, *, claimed: ClaimedMemoryConsolidation) -> _SummaryDraft:
        assert claimed.event.through_sequence == 2
        return _SummaryDraft(summary="摘要", topic="记忆主题")

    async def extract_candidates(self, *, scope: str, claimed: object, summary: object):
        del claimed, summary
        self.extract_scopes.append(scope)
        await asyncio.sleep(0)
        return (
            [_candidate("workspace", "language", 0.9), _candidate("workspace", "style", 0.8)]
            if scope == "workspace"
            else [_candidate("global", "global_language", 0.7)]
        )

    async def resolve_candidate(self, *, candidate: MemoryCandidate, matches: list[MemoryMatch]):
        self.resolved_keys.append(candidate.memory_key)
        if candidate.memory_key == "language":
            assert matches[0].memory_id == "existing-language"
            return MemoryAdjustmentBatch(
                adjustments=[
                    MemoryAdjustment(
                        memory_id="existing-language",
                        action="merge",
                        confidence_delta=0.1,
                    )
                ]
            )
        return MemoryAdjustmentBatch(
            adjustments=[MemoryAdjustment(memory_id="other", action="ignore_rel")]
        )


class _Repository:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve_memory_matches(self, *, candidate: MemoryCandidate, **_: object):
        self.queries.append(candidate.memory_key)
        await asyncio.sleep(0)
        return [
            MemoryMatch(
                memory_id="existing-language" if candidate.memory_key == "language" else "other",
                scope=candidate.scope,
                kind="policy",
                memory_key=candidate.memory_key,
                value={},
                display_text="existing",
                topics=[],
                confidence=0.8,
                effective_confidence=0.7,
                decay_rate=0.01,
                status="active",
                relevance=0.7,
            )
        ]


@pytest.mark.asyncio
async def test_consolidator_parallelizes_scope_and_candidate_work_with_a_batch_cap() -> None:
    model = _Model()
    repository = _Repository()
    consolidator = StructuredMemoryConsolidator(
        model=model,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        max_mutations=3,
    )

    claimed = _claimed()
    summary = await consolidator.summarize(claimed)
    result = await consolidator.consolidate(claimed, summary=summary)

    assert result.summary.topic == "记忆主题"
    assert model.extract_scopes == ["workspace", "global"]
    assert set(repository.queries) == {"language", "style", "global_language"}
    assert len(result.mutations) == 4
    merged_create = next(
        item
        for item in result.mutations
        if item.action == "create"
        and item.candidate is not None
        and item.candidate.memory_key == "language"
    )
    assert merged_create.candidate is not None
    assert merged_create.candidate.decay_rate == 0.002
    merged_deactivate = next(
        item
        for item in result.mutations
        if item.action == "deactivate"
        and item.target_memory_id == "existing-language"
    )
    assert merged_deactivate.target_memory_id == "existing-language"
    global_create = next(
        item
        for item in result.mutations
        if (
            item.action == "create"
            and item.candidate is not None
            and item.candidate.scope == "global"
        )
    )
    assert global_create.candidate is not None
    assert global_create.candidate.decay_rate == 0.0005


@pytest.mark.asyncio
async def test_consolidator_drops_malformed_resolution_without_mutation() -> None:
    class UnknownTargetModel(_Model):
        async def extract_candidates(self, *, scope: str, claimed: object, summary: object):
            del claimed, summary
            return [_candidate("workspace", "language", 0.9)] if scope == "workspace" else []

        async def resolve_candidate(
            self,
            *,
            candidate: MemoryCandidate,
            matches: list[MemoryMatch],
        ):
            del candidate, matches
            return MemoryAdjustmentBatch(
                adjustments=[MemoryAdjustment(memory_id="unknown", action="merge")]
            )

    consolidator = StructuredMemoryConsolidator(
        model=UnknownTargetModel(),  # type: ignore[arg-type]
        repository=_Repository(),  # type: ignore[arg-type]
        max_mutations=3,
    )

    claimed = _claimed()
    summary = await consolidator.summarize(claimed)
    result = await consolidator.consolidate(claimed, summary=summary)

    assert result.mutations == []


def test_deactivate_removes_each_explicit_conflicting_old_memory() -> None:
    candidate = _candidate("workspace", "export_template", 0.9)
    matches = [
        MemoryMatch(
            memory_id="old-template",
            scope="workspace",
            kind="policy",
            memory_key="export_template",
            value={},
            display_text="旧模板",
            topics=[],
            confidence=0.8,
            effective_confidence=0.8,
            decay_rate=0.01,
            status="active",
            relevance=0.5,
        ),
        MemoryMatch(
            memory_id="unrelated",
            scope="workspace",
            kind="policy",
            memory_key="language",
            value={},
            display_text="中文回复",
            topics=[],
            confidence=0.8,
            effective_confidence=0.8,
            decay_rate=0.01,
            status="active",
            relevance=0.5,
        ),
    ]
    mutations = _mutations_from_adjustments(
        candidate,
        matches,
        MemoryAdjustmentBatch(
            adjustments=[
                MemoryAdjustment(memory_id="old-template", action="deactivate"),
                MemoryAdjustment(memory_id="unrelated", action="ignore_rel"),
            ]
        ),
    )

    assert [item.action for item in mutations] == ["deactivate"]
    assert mutations[0].target_memory_id == "old-template"


def test_duplicate_restatement_still_merges_into_existing_memory() -> None:
    """只重复既有记忆的候选也按 merge 处理：新建合并版本并停用被合并的旧记录。"""

    candidate = _candidate("workspace", "reply_language", 0.9)
    matches = [
        MemoryMatch(
            memory_id="existing-language",
            scope="workspace",
            kind="policy",
            memory_key="reply_language",
            value={},
            display_text="项目回复默认使用中文",
            topics=[],
            confidence=0.8,
            effective_confidence=0.8,
            decay_rate=0.01,
            status="active",
            relevance=0.5,
        )
    ]
    mutations = _mutations_from_adjustments(
        candidate,
        matches,
        MemoryAdjustmentBatch(
            adjustments=[
                MemoryAdjustment(memory_id="existing-language", action="merge")
            ]
        ),
    )

    assert [item.action for item in mutations] == ["create", "deactivate"]
    assert mutations[1].target_memory_id == "existing-language"


@pytest.mark.asyncio
async def test_summary_generation_receives_existing_workspace_topics() -> None:
    class Client:
        request_messages: list[dict[str, str]]

        async def generate_structured(self, *, messages, **_: object):
            self.request_messages = messages
            return SimpleNamespace(summary="摘要", topic="已有主题")

    client = Client()
    model = DoubaoMemoryModel(client, max_topic_chars=32)  # type: ignore[arg-type]
    claimed = _claimed()
    claimed = ClaimedMemoryConsolidation(
        event=claimed.event,
        messages=claimed.messages,
        thread=claimed.thread,
        active_topics=[{"topic": "已有主题", "thread_count": 3}],
    )

    summary = await model.summarize(claimed=claimed)

    assert summary.topic == "已有主题"
    assert "active_workspace_topics" in client.request_messages[1]["content"]
    assert "已有主题" in client.request_messages[1]["content"]
