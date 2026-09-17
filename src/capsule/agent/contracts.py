"""Serializable contracts shared by the Agent runtime and its adapters."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    action: Literal["respond", "select_tools", "tool", "confirm", "finish"] = (
        "respond"
    )
    message: str | None = None
    # The first planner pass sees only the compact tool catalog.  It must use
    # this action to request complete schemas before it can create calls.
    selected_tool_names: list[str] = Field(default_factory=list, max_length=8)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=8)
    confirmation_message: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> PlanDecision:
        if self.action == "select_tools":
            if not self.selected_tool_names:
                raise ValueError("select_tools requires at least one selected tool")
            if self.tool_calls:
                raise ValueError("select_tools cannot contain tool calls")
        elif self.action in {"tool", "confirm"}:
            if not self.tool_calls:
                raise ValueError(f"{self.action} requires at least one tool call")
            if self.selected_tool_names:
                raise ValueError(
                    f"{self.action} must follow a prior select_tools decision"
                )
        elif self.selected_tool_names or self.tool_calls:
            raise ValueError(f"{self.action} cannot contain tool selection or calls")
        return self


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=128)
    # Reuse this value when the same client request is retried over the
    # network.  The server generates one when omitted.
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    graph_id: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    confirmation: bool | None = None
    cancel: bool = False
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
    pending_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_history: list[dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0


class AgentThreadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=255)


class AgentThreadRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    workspace_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)


class AgentThreadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_id: str
    workspace_id: str
    title: str
    status: str
    summary: str | None = None
    summary_topic: str | None = None
    memory_revision: int
    last_message_at: str | None = None
    deleted_at: str | None = None


class AgentMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    sequence: int
    role: str
    content: Any
    name: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    created_at: str
