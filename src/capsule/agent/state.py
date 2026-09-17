"""LangGraph state for one persisted conversational Agent thread."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

AgentStatus = Literal[
    "received",
    "planning",
    "tool_executed",
    "awaiting_confirmation",
    "completed",
    "cancelled",
    "failed",
    "max_steps",
]


class AgentState(TypedDict, total=False):
    thread_id: str
    # One complete Agent invocation/output. Multiple tool calls in the same
    # graph loop share this identifier and therefore count as one round.
    turn_id: str
    # Client-supplied identity for retrying one logical request.  The tool
    # layer can use this value when deriving an idempotency key.
    request_id: str
    # Runtime sets this only for the first graph invocation of a new round.
    # Internal tool-loop re-entry and confirmation resume keep it false.
    start_new_turn: bool
    user_id: str
    workspace_id: str
    graph_id: str | None
    granted_permissions: list[str]
    input_message: str | None
    confirmation_response: bool | None
    cancel_requested: bool
    messages: list[dict[str, Any]]
    memory_context: list[dict[str, Any]]
    # Memory is recalled once for each runtime request.  Tool-loop re-plans
    # reuse this exact set so an asynchronous writer cannot change context in
    # the middle of one response.
    memory_context_request_id: str | None
    working_context: dict[str, Any]
    # Tools follow progressive disclosure: the compact catalog is always
    # available, while complete schemas appear only for selected tools.
    tool_catalog: list[dict[str, Any]]
    tool_details: list[dict[str, Any]]
    deferred_tool_names: list[str]
    tool_disclosure_attempts: int
    context_budget: dict[str, Any]
    plan: dict[str, Any]
    pending_action: dict[str, Any] | None
    approved_action: dict[str, Any] | None
    tool_history: list[dict[str, Any]]
    # Calls planned for the current turn, with queued/running/terminal state.
    pending_tool_calls: list[dict[str, Any]]
    last_tool_results: list[dict[str, Any]]
    response: str | None
    status: AgentStatus
    step_count: int
    max_steps: int
    error: str | None
