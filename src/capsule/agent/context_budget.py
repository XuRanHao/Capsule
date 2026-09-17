"""Bounded, loss-aware context preparation before an Agent plans a response.

The module deliberately owns *projections* only.  PostgreSQL conversation
messages and tool-execution records remain the audit/source layer; context
preparation may replace an old message range with its already committed
summary, or reduce a tool result to a structured preview, but never deletes
the source data.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from capsule.db.agent_memory import ConversationContext, MemoryOutboxEvent

MemoryEventPublisher = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Model-input budget and the two context classes that may be reduced."""

    window_tokens: int = 32_000
    output_reserve_tokens: int = 4_000
    short_term_ratio: float = 0.30
    tool_result_ratio: float = 0.50
    raw_tail_tokens: int = 4_000
    max_reduction_rounds: int = 3
    summary_wait_seconds: float = 12.0
    summary_poll_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.window_tokens < 2:
            raise ValueError("window_tokens must be at least 2")
        if not 0 <= self.output_reserve_tokens < self.window_tokens:
            raise ValueError("output_reserve_tokens must fit inside window_tokens")
        if not 0 < self.short_term_ratio <= 1:
            raise ValueError("short_term_ratio must be in (0, 1]")
        if not 0 < self.tool_result_ratio <= 1:
            raise ValueError("tool_result_ratio must be in (0, 1]")
        if self.raw_tail_tokens < 1:
            raise ValueError("raw_tail_tokens must be positive")
        if self.max_reduction_rounds < 1:
            raise ValueError("max_reduction_rounds must be positive")
        if self.summary_wait_seconds <= 0 or self.summary_poll_seconds <= 0:
            raise ValueError("summary wait settings must be positive")

    @property
    def input_limit(self) -> int:
        return self.window_tokens - self.output_reserve_tokens

    @property
    def short_term_limit(self) -> int:
        return max(1, int(self.input_limit * self.short_term_ratio))

    @property
    def tool_result_limit(self) -> int:
        return max(1, int(self.input_limit * self.tool_result_ratio))


class ConversationRepository(Protocol):
    async def get_context(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        max_messages: int | None,
    ) -> ConversationContext: ...

    async def enqueue_consolidation(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        through_sequence: int,
    ) -> MemoryOutboxEvent | None: ...


@dataclass(frozen=True, slots=True)
class ContextTokenUsage:
    """A transparent accounting record, retained in the graph checkpoint."""

    total: int
    short_term: int
    tool_results: int
    long_term: int
    working: int
    tool_definitions: int

    def to_state(self, *, budget: ContextBudget, rounds: int) -> dict[str, int | bool]:
        return {
            "input_limit": budget.input_limit,
            "total_tokens": self.total,
            "short_term_tokens": self.short_term,
            "tool_result_tokens": self.tool_results,
            "long_term_tokens": self.long_term,
            "working_tokens": self.working,
            "tool_definition_tokens": self.tool_definitions,
            "reduction_rounds": rounds,
            "within_limit": self.total <= budget.input_limit,
        }


class ContextBudgetController:
    """Apply the agreed context policy at the LangGraph planning boundary.

    A short-term consolidation is requested only after the complete candidate
    context is too large *and* short-term context exceeds its share.  The
    request then waits solely for ``summary_covered_sequence``; the Worker's
    long/global memory extraction continues independently after that commit.
    """

    def __init__(
        self,
        *,
        budget: ContextBudget,
        repository: ConversationRepository | None = None,
        publish_event: MemoryEventPublisher | None = None,
    ) -> None:
        self._budget = budget
        self._repository = repository
        self._publish_event = publish_event

    async def govern(self, state: Mapping[str, object]) -> dict[str, object]:
        """Return a bounded context projection, or a deterministic failure."""

        updates: dict[str, object] = {}
        candidate = _copy_context_state(state)
        for round_index in range(1, self._budget.max_reduction_rounds + 1):
            usage = context_token_usage(candidate)
            if usage.total <= self._budget.input_limit:
                return _governed_updates(candidate, usage, self._budget, round_index - 1)

            should_summarize = (
                usage.short_term > self._budget.short_term_limit
                and self._repository is not None
            )
            should_compact_tools = usage.tool_results > self._budget.tool_result_limit
            summary_task: asyncio.Task[ConversationContext | None] | None = None
            if should_summarize:
                summary_task = asyncio.create_task(self._wait_for_short_summary(state))
            if should_compact_tools:
                candidate = _compact_tool_context(candidate)
            if summary_task is not None:
                refreshed = await summary_task
                if refreshed is not None:
                    candidate = _replace_short_term_context(candidate, refreshed)

            next_usage = context_token_usage(candidate)
            if next_usage.total < usage.total:
                continue
            # A summary may already be as small as it can get, and structured
            # tool previews are intentionally idempotent.  Finish with a
            # deterministic whole-item trim rather than looping forever.
            break

        candidate = _hard_trim_context(candidate, input_limit=self._budget.input_limit)
        usage = context_token_usage(candidate)
        updates.update(
            _governed_updates(
                candidate,
                usage,
                self._budget,
                self._budget.max_reduction_rounds,
            )
        )
        if usage.total > self._budget.input_limit:
            # Long/global memories and the current user input are immutable in
            # this request.  If they alone do not fit, invoking the model would
            # be less predictable than returning an explicit bounded failure.
            updates["response"] = "当前输入及不可压缩上下文超出模型限制，请缩短本次输入。"
            updates["status"] = "failed"
            updates["error"] = "context_budget_exceeded"
        return updates

    async def _wait_for_short_summary(
        self,
        state: Mapping[str, object],
    ) -> ConversationContext | None:
        repository = self._repository
        if repository is None:
            return None
        context = await repository.get_context(
            thread_id=str(state["thread_id"]),
            user_id=str(state["user_id"]),
            workspace_id=str(state["workspace_id"]),
            max_messages=None,
        )
        through_sequence = _summary_target_sequence(
            context,
            raw_tail_tokens=self._budget.raw_tail_tokens,
        )
        if through_sequence is None:
            return None
        event = await repository.enqueue_consolidation(
            thread_id=str(state["thread_id"]),
            user_id=str(state["user_id"]),
            workspace_id=str(state["workspace_id"]),
            through_sequence=through_sequence,
        )
        if event is not None and self._publish_event is not None:
            await self._publish_event(event)

        deadline = monotonic() + self._budget.summary_wait_seconds
        while True:
            refreshed = await repository.get_context(
                thread_id=str(state["thread_id"]),
                user_id=str(state["user_id"]),
                workspace_id=str(state["workspace_id"]),
                max_messages=None,
            )
            if refreshed.thread.summary_covered_sequence >= through_sequence:
                return refreshed
            if monotonic() >= deadline:
                return None
            await asyncio.sleep(self._budget.summary_poll_seconds)


def context_token_usage(state: Mapping[str, object]) -> ContextTokenUsage:
    """Estimate only fields exposed to a planner; durable audit data is excluded."""

    messages = _as_list(state.get("messages"))
    short_messages = [item for item in messages if item.get("role") != "tool"]
    tool_messages = [item for item in messages if item.get("role") == "tool"]
    working_context = _as_mapping(state.get("working_context"))
    short_summary = {
        key: working_context.get(key)
        for key in ("conversation_summary", "conversation_topic")
        if working_context.get(key) is not None
    }
    working = {
        key: value
        for key, value in working_context.items()
        if key not in {"conversation_summary", "conversation_topic"}
    }
    short_term = _tokens(short_messages) + _tokens(short_summary)
    tool_results = _tokens(tool_messages) + _tokens(_as_list(state.get("tool_history")))
    long_term = _tokens(_as_list(state.get("memory_context")))
    tool_definitions = _tokens(_as_list(state.get("tool_catalog"))) + _tokens(
        _as_list(state.get("tool_details"))
    )
    working_tokens = _tokens(working)
    return ContextTokenUsage(
        total=short_term + tool_results + long_term + tool_definitions + working_tokens,
        short_term=short_term,
        tool_results=tool_results,
        long_term=long_term,
        working=working_tokens,
        tool_definitions=tool_definitions,
    )


def _copy_context_state(state: Mapping[str, object]) -> dict[str, object]:
    return {
        "messages": [dict(item) for item in _as_list(state.get("messages"))],
        "tool_history": [dict(item) for item in _as_list(state.get("tool_history"))],
        "working_context": dict(_as_mapping(state.get("working_context"))),
        "memory_context": [dict(item) for item in _as_list(state.get("memory_context"))],
        "tool_catalog": [dict(item) for item in _as_list(state.get("tool_catalog"))],
        "tool_details": [dict(item) for item in _as_list(state.get("tool_details"))],
        "deferred_tool_names": _as_strings(state.get("deferred_tool_names")),
    }


def _governed_updates(
    candidate: Mapping[str, object],
    usage: ContextTokenUsage,
    budget: ContextBudget,
    rounds: int,
) -> dict[str, object]:
    return {
        "messages": candidate["messages"],
        "tool_history": candidate["tool_history"],
        "working_context": candidate["working_context"],
        "tool_details": candidate["tool_details"],
        "deferred_tool_names": candidate["deferred_tool_names"],
        "context_budget": usage.to_state(budget=budget, rounds=rounds),
    }


def _replace_short_term_context(
    candidate: Mapping[str, object],
    context: ConversationContext,
) -> dict[str, object]:
    refreshed = dict(candidate)
    # Tool output is not chat history.  It remains available for the current
    # tool loop (and is independently compacted below) while durable user and
    # assistant context is rebuilt from the newly committed summary watermark.
    refreshed["messages"] = [
        *[item.to_graph_message() for item in context.messages],
        *[
            dict(item)
            for item in _as_list(candidate.get("messages"))
            if item.get("role") == "tool"
        ],
    ]
    refreshed["working_context"] = {
        **_as_mapping(candidate.get("working_context")),
        "conversation_summary": context.thread.summary,
        "conversation_topic": context.thread.summary_topic,
        "summary_covered_sequence": context.thread.summary_covered_sequence,
        "memory_revision": context.thread.memory_revision,
        # A compacted view is authoritative for the next planning step even if
        # the hot mirror will be refreshed again after this invocation.
        "hot_mirror_through_sequence": context.thread.last_message_sequence,
    }
    return refreshed


def _summary_target_sequence(
    context: ConversationContext,
    *,
    raw_tail_tokens: int,
) -> int | None:
    """Leave a recent raw tail and return the older prefix safe to summarize."""

    if not context.messages:
        return None
    tail_tokens = 0
    first_tail_index = len(context.messages)
    for index in range(len(context.messages) - 1, -1, -1):
        message = context.messages[index]
        # Always retain the current/latest source message even when it itself
        # is large; user input is never silently rewritten by this component.
        if (
            index < len(context.messages) - 1
            and tail_tokens + message.estimated_tokens > raw_tail_tokens
        ):
            break
        tail_tokens += message.estimated_tokens
        first_tail_index = index
    if first_tail_index == 0:
        return None
    target = context.messages[first_tail_index - 1].sequence
    if target <= context.thread.summary_covered_sequence:
        return None
    return target


def _compact_tool_context(candidate: Mapping[str, object]) -> dict[str, object]:
    compacted = dict(candidate)
    compacted["tool_history"] = [
        _compact_tool_result(item) for item in _as_list(candidate.get("tool_history"))
    ]
    messages: list[dict[str, Any]] = []
    for item in _as_list(candidate.get("messages")):
        if item.get("role") != "tool":
            messages.append(dict(item))
            continue
        messages.append(
            {
                **{key: value for key, value in item.items() if key != "content"},
                "content": _compact_value(item.get("content")),
                "context_projection": "structured_preview",
            }
        )
    compacted["messages"] = messages
    return compacted


def _compact_tool_result(item: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: item[key]
        for key in (
            "call_id",
            "operation_id",
            "name",
            "turn_id",
            "ok",
            "needs_confirmation",
            "error_code",
            "error_message",
        )
        if key in item and item[key] is not None
    }
    if item.get("ok"):
        result["output"] = _compact_value(item.get("output"))
    return result


def _compact_value(value: Any, *, string_limit: int = 600, item_limit: int = 8) -> Any:
    """Keep stable identifiers/outcomes while bounding arbitrary raw output."""

    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return {"preview": value[:string_limit], "truncated_chars": len(value) - string_limit}
    if isinstance(value, list):
        kept = [_compact_value(item) for item in value[:item_limit]]
        if len(value) > item_limit:
            kept.append({"omitted_items": len(value) - item_limit})
        return kept
    if isinstance(value, Mapping):
        important = ("id", "ids", "status", "error", "message", "result", "url", "name")
        keys = list(value)
        preferred = [key for key in important if key in value]
        selected = list(dict.fromkeys(preferred + keys[:item_limit]))
        preview = {str(key): _compact_value(value[key]) for key in selected[:item_limit]}
        if len(keys) > len(selected[:item_limit]):
            preview["omitted_fields"] = len(keys) - len(selected[:item_limit])
        return preview
    return value


def _hard_trim_context(
    candidate: Mapping[str, object],
    *,
    input_limit: int,
) -> dict[str, object]:
    """Drop oldest whole projections; never alter current input or recalled memory."""

    trimmed = _compact_tool_context(candidate)
    messages = [dict(item) for item in _as_list(trimmed.get("messages"))]
    history = [dict(item) for item in _as_list(trimmed.get("tool_history"))]
    details = [dict(item) for item in _as_list(trimmed.get("tool_details"))]
    deferred = _as_strings(trimmed.get("deferred_tool_names"))
    while _trimmed_usage(trimmed, messages, history, details).total > input_limit:
        tool_index = next(
            (index for index, item in enumerate(messages) if item.get("role") == "tool"),
            None,
        )
        if tool_index is not None:
            messages.pop(tool_index)
            if history:
                history.pop(0)
            continue
        # Full schemas are removed as a unit.  A planner can select them again
        # in a later disclosure step; no selected schema is text-truncated.
        if len(details) > 1:
            deferred.append(str(details.pop().get("name", "")))
            continue
        latest_non_tool = max(
            (index for index, item in enumerate(messages) if item.get("role") != "tool"),
            default=-1,
        )
        removable = next(
            (
                index
                for index, item in enumerate(messages)
                if index != latest_non_tool and item.get("role") != "tool"
            ),
            None,
        )
        if removable is None:
            break
        messages.pop(removable)
    trimmed["messages"] = messages
    trimmed["tool_history"] = history
    trimmed["tool_details"] = details
    trimmed["deferred_tool_names"] = list(dict.fromkeys(name for name in deferred if name))
    return trimmed


def _trimmed_usage(
    candidate: Mapping[str, object],
    messages: list[dict[str, Any]],
    history: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> ContextTokenUsage:
    return context_token_usage(
        {
            **candidate,
            "messages": messages,
            "tool_history": history,
            "tool_details": details,
        }
    )


def _tokens(value: Any) -> int:
    if value in ({}, [], None):
        return 0
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, (len(encoded) + 3) // 4)


def _as_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _as_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_strings(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
