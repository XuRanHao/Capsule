"""Serializable contracts shared by the Agent runtime and its adapters."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str = Field(default_factory=lambda: f"call_{uuid4().hex}", min_length=1)
    operation_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=255)
    value: Any


class PlanDecision(BaseModel):
    """Planner output; keeping it structured makes the graph model-independent."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["respond", "tool", "confirm", "finish"] = "respond"
    message: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    memory_writes: list[MemoryWrite] = Field(default_factory=list, max_length=20)
    confirmation_message: str | None = None


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    graph_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    confirmation: bool | None = None
    max_steps: int = Field(default=8, ge=1, le=32)


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    status: Literal[
        "completed",
        "awaiting_confirmation",
        "cancelled",
        "failed",
        "max_steps",
    ]
    message: str | None = None
    pending_action: dict[str, Any] | None = None
    tool_history: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
