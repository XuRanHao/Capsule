import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from capsule.agent.memory_consolidator import StructuredMemoryConsolidator
from capsule.agent.memory_contracts import MemoryCandidate
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


@dataclass
class _Resolution:
    action: str
    target_memory_id: str | None = None
    confidence_delta: float = 0.0


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
            return _Resolution("merge", "existing-language", 0.1)
        return _Resolution("create")


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

    result = await consolidator.consolidate(_claimed())

    assert result.summary.topic == "记忆主题"
    assert model.extract_scopes == ["workspace", "global"]
    assert set(repository.queries) == {"language", "style", "global_language"}
    assert len(result.mutations) == 3
    merge = next(item for item in result.mutations if item.action == "merge")
    assert merge.target_memory_id == "existing-language"
    assert merge.confidence_delta == 0.1
    assert merge.candidate is not None
    assert merge.candidate.decay_rate == 0.002
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
async def test_consolidator_falls_back_to_create_when_resolution_targets_unknown_memory() -> None:
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
            return _Resolution("merge", "unknown", 0.2)

    consolidator = StructuredMemoryConsolidator(
        model=UnknownTargetModel(),  # type: ignore[arg-type]
        repository=_Repository(),  # type: ignore[arg-type]
        max_mutations=3,
    )

    result = await consolidator.consolidate(_claimed())

    assert len(result.mutations) == 1
    assert result.mutations[0].action == "create"
