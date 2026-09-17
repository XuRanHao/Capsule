from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from capsule.agent.context_budget import ContextBudget, ContextBudgetController
from capsule.db.agent_memory import (
    AgentMessageRecord,
    AgentThreadRecord,
    ConversationContext,
    MemoryOutboxEvent,
)


def _thread(*, covered_sequence: int = 0) -> AgentThreadRecord:
    return AgentThreadRecord(
        thread_id="thread-1",
        user_id="user-1",
        workspace_id="workspace-1",
        title="上下文测试",
        status="active",
        summary="旧摘要" if covered_sequence else None,
        summary_topic="旧主题" if covered_sequence else None,
        summary_covered_sequence=covered_sequence,
        last_consolidated_sequence=covered_sequence,
        memory_revision=covered_sequence,
        last_message_at=datetime.now(UTC),
        last_message_sequence=3,
    )


def _message(sequence: int, content: str) -> AgentMessageRecord:
    return AgentMessageRecord(
        message_id=f"message-{sequence}",
        thread_id="thread-1",
        sequence=sequence,
        role="user" if sequence % 2 else "assistant",
        content=content,
        name=None,
        turn_id=f"turn-{sequence}",
        request_id=f"request-{sequence}",
        estimated_tokens=max(1, len(content) // 4),
        created_at=datetime.now(UTC),
    )


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "thread_id": "thread-1",
        "user_id": "user-1",
        "workspace_id": "workspace-1",
        "messages": [{"role": "user", "content": "当前输入"}],
        "tool_history": [],
        "working_context": {},
        "memory_context": [],
        "tool_catalog": [],
        "tool_details": [],
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_context_budget_does_not_start_memory_work_before_total_overflow() -> None:
    controller = ContextBudgetController(
        budget=ContextBudget(window_tokens=1_000, output_reserve_tokens=100)
    )

    updates = await controller.govern(_state())

    assert updates["messages"] == [{"role": "user", "content": "当前输入"}]
    assert updates["context_budget"]["reduction_rounds"] == 0  # type: ignore[index]


@pytest.mark.asyncio
async def test_context_budget_compacts_only_tool_result_projection() -> None:
    recalled = [{"memory_id": "memory-1", "display_text": "长期记忆必须完整保留。"}]
    controller = ContextBudgetController(
        budget=ContextBudget(
            window_tokens=1_000,
            output_reserve_tokens=100,
            tool_result_ratio=0.2,
        )
    )
    state = _state(
        messages=[
            {"role": "user", "content": "当前输入"},
            {"role": "tool", "name": "read", "content": "x" * 4_000},
        ],
        tool_history=[
            {
                "call_id": "call-1",
                "name": "read",
                "ok": True,
                "output": "x" * 4_000,
            }
        ],
        memory_context=recalled,
    )

    updates = await controller.govern(state)

    tool_message = updates["messages"][1]  # type: ignore[index]
    assert tool_message["context_projection"] == "structured_preview"
    assert isinstance(tool_message["content"], dict)
    assert updates["tool_history"][0]["output"] != "x" * 4_000  # type: ignore[index]
    # The controller never rewrites recalled long/global memory records.
    assert state["memory_context"] == recalled


@pytest.mark.asyncio
async def test_context_budget_waits_only_for_committed_short_summary() -> None:
    before = ConversationContext(
        thread=_thread(),
        messages=[_message(1, "a" * 800), _message(2, "b" * 800), _message(3, "当前输入")],
    )
    after = ConversationContext(
        thread=replace(
            _thread(covered_sequence=2),
            summary="新摘要",
            summary_topic="新主题",
        ),
        messages=[_message(3, "当前输入")],
    )

    class Repository:
        def __init__(self) -> None:
            self.current = before
            self.enqueued: list[int] = []

        async def get_context(self, **_: object) -> ConversationContext:
            return self.current

        async def enqueue_consolidation(
            self,
            *,
            through_sequence: int,
            **_: object,
        ) -> MemoryOutboxEvent:
            self.enqueued.append(through_sequence)
            return MemoryOutboxEvent(
                event_id="event-1",
                thread_id="thread-1",
                user_id="user-1",
                workspace_id="workspace-1",
                through_sequence=through_sequence,
            )

    repository = Repository()

    async def publish(_: MemoryOutboxEvent) -> None:
        # This emulates the Worker's first transaction.  Its long/global
        # extraction is intentionally not part of the request's wait point.
        repository.current = after

    controller = ContextBudgetController(
        budget=ContextBudget(
            window_tokens=450,
            output_reserve_tokens=50,
            short_term_ratio=0.3,
            raw_tail_tokens=100,
        ),
        repository=repository,
        publish_event=publish,
    )

    updates = await controller.govern(
        _state(messages=[item.to_graph_message() for item in before.messages])
    )

    assert repository.enqueued == [2]
    assert updates["working_context"]["conversation_summary"] == "新摘要"  # type: ignore[index]
    assert [item["content"] for item in updates["messages"]] == ["当前输入"]  # type: ignore[index]
