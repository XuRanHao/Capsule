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
    user_id: str
    workspace_id: str
    graph_id: str | None
    granted_permissions: list[str]
    input_message: str | None
    confirmation_response: bool | None
    messages: list[dict[str, Any]]
    memory_context: list[dict[str, Any]]
    working_context: dict[str, Any]
    plan: dict[str, Any]
    pending_action: dict[str, Any] | None
    approved_action: dict[str, Any] | None
    tool_history: list[dict[str, Any]]
    last_tool_results: list[dict[str, Any]]
    response: str | None
    status: AgentStatus
    step_count: int
    max_steps: int
    error: str | None
