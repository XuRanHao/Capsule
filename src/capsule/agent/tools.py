"""Validated, bounded tool execution for the narrative Agent."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

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
    timeout_seconds: float = 5.0
    max_attempts: int = 1
    requires_confirmation: bool = False
    required_permission: str | None = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    operation_id: str | None = None
    name: str
    ok: bool
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = Field(default=0, ge=0)
    needs_confirmation: bool = False


class ToolExecutionStore(Protocol):
    async def create_operation(self, **kwargs: Any) -> str: ...

    async def get_operation(self, **kwargs: Any) -> dict[str, Any] | None: ...

    async def update_operation(self, **kwargs: Any) -> bool: ...

    async def list_for_thread(self, **kwargs: Any) -> list[dict[str, Any]]: ...


class ToolRegistry:
    """Server-side tool registry; the model only selects registered names."""

    def __init__(
        self,
        tools: list[AgentTool] | None = None,
        execution_store: ToolExecutionStore | None = None,
    ) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._execution_store = execution_store
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
        operation_id, audit_error, existing = await self._start_operation(call, tool, context)
        if audit_error is not None:
            return ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="execution_audit_failed",
                error_message=audit_error,
            )
        if operation_id is None:
            operation_id = f"local_{uuid4().hex}"
        if tool is None:
            result = ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="unknown_tool",
                error_message="tool is not registered",
            )
            await self._finish_operation(result, context)
            return result
        if tool.required_permission is not None and not _has_permission(
            context.granted_permissions, tool.required_permission
        ):
            result = ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="permission_denied",
                error_message="the current session is not allowed to use this tool",
            )
            await self._finish_operation(result, context)
            return result
        if existing is not None and existing["execution_status"] == "succeeded":
            return ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=True,
                output=existing.get("output"),
                attempts=int(existing.get("attempts", 1)),
            )
        if tool.requires_confirmation and not confirmed:
            if self._execution_store is not None:
                await self._execution_store.update_operation(
                    operation_id=operation_id,
                    user_id=context.user_id,
                    workspace_id=context.workspace_id,
                    thread_id=context.thread_id,
                    execution_status="awaiting_confirmation",
                    confirmation_status="pending",
                )
            return ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="confirmation_required",
                error_message="tool execution requires user confirmation",
                needs_confirmation=True,
            )
        try:
            arguments = tool.args_schema.model_validate(call.arguments)
        except ValidationError as exc:
            result = ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="invalid_arguments",
                error_message=str(exc)[:2000],
            )
            await self._finish_operation(result, context)
            return result

        last_error: str | None = None
        for attempt in range(1, tool.max_attempts + 1):
            if self._execution_store is not None:
                await self._execution_store.update_operation(
                    operation_id=operation_id,
                    user_id=context.user_id,
                    workspace_id=context.workspace_id,
                    thread_id=context.thread_id,
                    execution_status="running",
                    confirmation_status="confirmed"
                    if tool.requires_confirmation
                    else "not_required",
                    attempts=attempt,
                    started_at=datetime.now(UTC),
                )
            try:
                value = tool.handler(arguments, context)
                if inspect.isawaitable(value):
                    value = await asyncio.wait_for(value, timeout=tool.timeout_seconds)
                result = ToolExecutionResult(
                    call_id=call.call_id,
                    operation_id=operation_id,
                    name=call.name,
                    ok=True,
                    output=value,
                    attempts=attempt,
                )
                try:
                    await self._finish_operation(result, context)
                except Exception as exc:
                    return ToolExecutionResult(
                        call_id=call.call_id,
                        operation_id=operation_id,
                        name=call.name,
                        ok=False,
                        error_code="execution_audit_failed",
                        error_message=str(exc) or type(exc).__name__,
                        attempts=attempt,
                    )
                return result
            except TimeoutError:
                last_error = f"tool timed out after {tool.timeout_seconds:.1f}s"
            except Exception as exc:  # tool failures become planner-visible data
                last_error = str(exc) or type(exc).__name__
        result = ToolExecutionResult(
            call_id=call.call_id,
            operation_id=operation_id,
            name=call.name,
            ok=False,
            error_code="execution_failed",
            error_message=last_error,
            attempts=tool.max_attempts,
        )
        await self._finish_operation(result, context)
        return result

    async def _start_operation(
        self,
        call: ToolCall,
        tool: AgentTool | None,
        context: ToolContext,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        if self._execution_store is None:
            return call.operation_id, None, None
        try:
            if call.operation_id:
                existing = await self._execution_store.get_operation(
                    operation_id=call.operation_id,
                    user_id=context.user_id,
                    workspace_id=context.workspace_id,
                    thread_id=context.thread_id,
                )
                if existing is not None:
                    return call.operation_id, None, existing
            operation_id = await self._execution_store.create_operation(
                operation_id=call.operation_id,
                call_id=call.call_id,
                thread_id=context.thread_id,
                user_id=context.user_id,
                workspace_id=context.workspace_id,
                graph_id=context.graph_id,
                tool_name=call.name,
                required_permission=tool.required_permission if tool else None,
                arguments=call.arguments,
                confirmation_status="not_required",
                execution_status="created",
            )
            return operation_id, None, None
        except Exception as exc:
            return call.operation_id, str(exc) or type(exc).__name__, None

    async def _finish_operation(
        self,
        result: ToolExecutionResult,
        context: ToolContext,
    ) -> None:
        if self._execution_store is None or result.operation_id is None:
            return
        await self._execution_store.update_operation(
            operation_id=result.operation_id,
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            thread_id=context.thread_id,
            execution_status="succeeded" if result.ok else "failed",
            attempts=result.attempts,
            output=result.output,
            error_code=result.error_code,
            error_message=result.error_message,
            finished_at=datetime.now(UTC),
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
