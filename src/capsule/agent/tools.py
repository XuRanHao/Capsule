"""Validated, bounded tool execution for the narrative Agent."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from capsule.agent.contracts import ToolCall


@dataclass(frozen=True, slots=True)
class ToolContext:
    user_id: str
    workspace_id: str
    thread_id: str
    graph_id: str | None
    state: Mapping[str, Any]
    granted_permissions: frozenset[str] = frozenset()


ToolHandler = Callable[[BaseModel, ToolContext], Awaitable[Any] | Any]


@dataclass(frozen=True, slots=True)
class AgentTool:
    name: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    timeout_seconds: float = 30.0
    max_attempts: int = 1
    requires_confirmation: bool = False
    required_permission: str | None = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    ok: bool
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = Field(default=0, ge=0)
    needs_confirmation: bool = False


class ToolRegistry:
    """Server-side tool registry; the model only selects registered names."""

    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"duplicate or empty Agent tool name: {tool.name!r}")
        if tool.timeout_seconds <= 0 or tool.max_attempts < 1:
            raise ValueError("tool timeout_seconds must be positive and max_attempts >= 1")
        if tool.required_permission is not None and not tool.required_permission.strip():
            raise ValueError("tool required_permission must be non-empty when provided")
        self._tools[tool.name] = tool

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_confirmation": tool.requires_confirmation,
                "required_permission": tool.required_permission,
                "args_schema": tool.args_schema.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    async def execute(
        self,
        call: ToolCall,
        *,
        context: ToolContext,
        confirmed: bool = False,
    ) -> ToolExecutionResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error_code="unknown_tool",
                error_message="tool is not registered",
            )
        if tool.required_permission is not None and not _has_permission(
            context.granted_permissions, tool.required_permission
        ):
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error_code="permission_denied",
                error_message="the current session is not allowed to use this tool",
            )
        if tool.requires_confirmation and not confirmed:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error_code="confirmation_required",
                error_message="tool execution requires user confirmation",
                needs_confirmation=True,
            )
        try:
            arguments = tool.args_schema.model_validate(call.arguments)
        except ValidationError as exc:
            return ToolExecutionResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error_code="invalid_arguments",
                error_message=str(exc)[:2000],
            )

        last_error: str | None = None
        for attempt in range(1, tool.max_attempts + 1):
            try:
                value = tool.handler(arguments, context)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=tool.timeout_seconds)
                return ToolExecutionResult(
                    call_id=call.call_id,
                    name=call.name,
                    ok=True,
                    output=value,
                    attempts=attempt,
                )
            except TimeoutError:
                last_error = f"tool timed out after {tool.timeout_seconds:.1f}s"
            except Exception as exc:  # tool failures become planner-visible data
                last_error = str(exc) or type(exc).__name__
        return ToolExecutionResult(
            call_id=call.call_id,
            name=call.name,
            ok=False,
            error_code="execution_failed",
            error_message=last_error,
            attempts=tool.max_attempts,
        )


def _has_permission(granted: frozenset[str], required: str) -> bool:
    """Apply the small server-side graph permission hierarchy."""

    if "*" in granted or "graph:admin" in granted:
        return True
    if required in granted:
        return True
    if required == "graph:read":
        return "graph:write" in granted or "graph:destructive" in granted
    if required == "graph:write":
        return "graph:destructive" in granted
    return False
