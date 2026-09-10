"""Validated, bounded tool execution for the narrative Agent."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
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
    turn_id: str = ""
    request_id: str = ""


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
    output_schema: type[BaseModel] | None = None
    max_output_bytes: int = 64 * 1024
    concurrency_mode: Literal["parallel", "exclusive"] = "exclusive"
    lock_scope: Literal["none", "graph", "entity", "asset"] = "graph"


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


ToolPreHook = Callable[
    [ToolCall, AgentTool | None, ToolContext],
    Awaitable[ToolExecutionResult | None] | ToolExecutionResult | None,
]
ToolPostHook = Callable[
    [ToolExecutionResult, ToolCall, AgentTool | None, ToolContext],
    Awaitable[ToolExecutionResult | None] | ToolExecutionResult | None,
]


@dataclass(frozen=True, slots=True)
class ToolHooks:
    """Server-side hooks shared by every registered tool execution."""

    before: tuple[ToolPreHook, ...] = ()
    after: tuple[ToolPostHook, ...] = ()


class ToolExecutionStore(Protocol):
    supports_atomic_claim: bool

    async def create_operation(self, **kwargs: Any) -> str: ...

    async def claim_operation(self, **kwargs: Any) -> dict[str, Any]: ...

    async def get_operation(self, **kwargs: Any) -> dict[str, Any] | None: ...

    async def update_operation(self, **kwargs: Any) -> bool: ...

    async def list_for_thread(self, **kwargs: Any) -> list[dict[str, Any]]: ...


class ToolRegistry:
    """Server-side tool registry; the model only selects registered names."""

    def __init__(
        self,
        tools: list[AgentTool] | None = None,
        execution_store: ToolExecutionStore | None = None,
        hooks: ToolHooks | None = None,
    ) -> None:
        self._tools: dict[str, AgentTool] = {}
        self._execution_store = execution_store
        self._hooks = hooks or ToolHooks()
        self._worker_id = f"worker_{uuid4().hex}"
        self._locks: dict[str, asyncio.Lock] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"duplicate or empty Agent tool name: {tool.name!r}")
        if tool.timeout_seconds <= 0 or tool.max_attempts < 1:
            raise ValueError("tool timeout_seconds must be positive and max_attempts >= 1")
        if tool.required_permission is not None and not tool.required_permission.strip():
            raise ValueError("tool required_permission must be non-empty when provided")
        if tool.output_schema is not None:
            try:
                is_model = issubclass(tool.output_schema, BaseModel)
            except TypeError:
                is_model = False
            if not is_model:
                raise ValueError("tool output_schema must be a Pydantic BaseModel")
        if tool.max_output_bytes <= 0:
            raise ValueError("tool max_output_bytes must be positive")
        if tool.concurrency_mode not in {"parallel", "exclusive"}:
            raise ValueError("tool concurrency_mode must be parallel or exclusive")
        if tool.lock_scope not in {"none", "graph", "entity", "asset"}:
            raise ValueError("tool lock_scope must be none, graph, entity, or asset")
        self._tools[tool.name] = tool

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "requires_confirmation": tool.requires_confirmation,
                "required_permission": tool.required_permission,
                "args_schema": tool.args_schema.model_json_schema(),
                "output_schema": (
                    tool.output_schema.model_json_schema()
                    if tool.output_schema is not None
                    else None
                ),
                "max_output_bytes": tool.max_output_bytes,
                "concurrency_mode": tool.concurrency_mode,
                "lock_scope": tool.lock_scope,
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
        try:
            pre_result = await self._run_before_hooks(call, tool, context)
        except Exception as exc:
            result = ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="pre_hook_failed",
                error_message=str(exc) or type(exc).__name__,
            )
            await self._finish_operation(result, context)
            return result
        if pre_result is not None:
            if pre_result.operation_id is None:
                pre_result = pre_result.model_copy(update={"operation_id": operation_id})
            await self._finish_operation(pre_result, context)
            return pre_result
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
        claim = await self._claim_operation(operation_id, context)
        if claim is not None and not claim.get("claimed", False):
            if claim.get("reason") == "already_succeeded":
                return ToolExecutionResult(
                    call_id=call.call_id,
                    operation_id=operation_id,
                    name=call.name,
                    ok=True,
                    output=claim.get("output"),
                    attempts=int(claim.get("attempts", 1)),
                )
            return ToolExecutionResult(
                call_id=call.call_id,
                operation_id=operation_id,
                name=call.name,
                ok=False,
                error_code="operation_in_progress"
                if claim.get("reason") == "in_progress"
                else "execution_claim_failed",
                error_message="the logical tool operation is already being executed"
                if claim.get("reason") == "in_progress"
                else "the tool operation could not be claimed",
            )
        if tool.concurrency_mode == "exclusive":
            lock = self._get_lock(tool, context, arguments)
            async with lock:
                result = await self._run_with_retries(
                    call, tool, arguments, context, operation_id
                )
        else:
            result = await self._run_with_retries(
                call, tool, arguments, context, operation_id
            )
        return await self._complete_operation(result, call, tool, context)

    async def cancel_operation(self, operation_id: str, context: ToolContext) -> bool:
        """Cancel a queued operation without exposing the store to the graph."""

        if self._execution_store is None:
            return False
        return await self._execution_store.update_operation(
            operation_id=operation_id,
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            thread_id=context.thread_id,
            execution_status="cancelled",
            confirmation_status="rejected",
            error_code="operation_cancelled",
            error_message="operation cancelled before completion",
            finished_at=datetime.now(UTC),
            lease_owner=None,
            lease_expires_at=None,
        )

    async def _run_with_retries(
        self,
        call: ToolCall,
        tool: AgentTool,
        arguments: BaseModel,
        context: ToolContext,
        operation_id: str,
    ) -> ToolExecutionResult:
        """Run one tool call, keeping an exclusive lock across its retries."""

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
                if inspect.iscoroutinefunction(tool.handler):
                    value = tool.handler(arguments, context)
                    value = await self._await_with_timeout(value, tool.timeout_seconds)
                else:
                    value = await asyncio.wait_for(
                        asyncio.to_thread(tool.handler, arguments, context),
                        timeout=tool.timeout_seconds,
                    )
                    if inspect.isawaitable(value):
                        value = await self._await_with_timeout(value, tool.timeout_seconds)
                return ToolExecutionResult(
                    call_id=call.call_id,
                    operation_id=operation_id,
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
            operation_id=operation_id,
            name=call.name,
            ok=False,
            error_code="execution_failed",
            error_message=last_error,
            attempts=tool.max_attempts,
        )

    @staticmethod
    async def _await_with_timeout(value: Any, timeout_seconds: float) -> Any:
        try:
            return await asyncio.wait_for(value, timeout=timeout_seconds)
        except TimeoutError:
            # A synchronous wrapper may return a coroutine object. Close it
            # when the bounded wait is cancelled so it is not leaked.
            if inspect.iscoroutine(value):
                value.close()
            raise

    def _get_lock(
        self,
        tool: AgentTool,
        context: ToolContext,
        arguments: BaseModel,
    ) -> asyncio.Lock:
        """Return the process-local lock for an exclusive tool resource."""

        values = arguments.model_dump(mode="json")
        scope = tool.lock_scope
        if scope == "graph":
            resource_id = getattr(context, "graph_id", None) or "unknown"
            key = f"graph:{resource_id}"
        elif scope == "entity":
            resource_id = _first_argument_value(
                values,
                "entity_id",
                "source_entity_id",
                "target_entity_id",
            )
            key = f"entity:{resource_id or tool.name}"
        elif scope == "asset":
            resource_id = _first_argument_value(values, "asset_id")
            key = f"asset:{resource_id or tool.name}"
        else:
            key = f"tool:{tool.name}"
        key = f"{getattr(context, 'workspace_id', 'default')}:{key}"
        return self._locks.setdefault(key, asyncio.Lock())

    async def _run_before_hooks(
        self,
        call: ToolCall,
        tool: AgentTool,
        context: ToolContext,
    ) -> ToolExecutionResult | None:
        for hook in self._hooks.before:
            result = hook(call, tool, context)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return result
        return None

    async def _run_after_hooks(
        self,
        result: ToolExecutionResult,
        call: ToolCall,
        tool: AgentTool,
        context: ToolContext,
    ) -> ToolExecutionResult:
        current = result
        for hook in self._hooks.after:
            updated = hook(current, call, tool, context)
            if inspect.isawaitable(updated):
                updated = await updated
            if updated is not None:
                current = updated
        return current

    async def _complete_operation(
        self,
        result: ToolExecutionResult,
        call: ToolCall,
        tool: AgentTool,
        context: ToolContext,
    ) -> ToolExecutionResult:
        if result.ok:
            result = self._validate_output(result, tool)
        try:
            result = await self._run_after_hooks(result, call, tool, context)
        except Exception as exc:
            result = ToolExecutionResult(
                call_id=result.call_id,
                operation_id=result.operation_id,
                name=result.name,
                ok=False,
                error_code="post_hook_failed",
                error_message=str(exc) or type(exc).__name__,
                attempts=result.attempts,
            )
        if result.ok:
            result = self._validate_output(result, tool)
        try:
            await self._finish_operation(result, context)
        except Exception as exc:
            return ToolExecutionResult(
                call_id=result.call_id,
                operation_id=result.operation_id,
                name=result.name,
                ok=False,
                error_code="execution_audit_failed",
                error_message=str(exc) or type(exc).__name__,
                attempts=result.attempts,
            )
        return result

    def _validate_output(
        self,
        result: ToolExecutionResult,
        tool: AgentTool,
    ) -> ToolExecutionResult:
        """Normalize a tool result and enforce its output contract and size."""

        try:
            output = result.output
            if tool.output_schema is not None:
                output = tool.output_schema.model_validate(output).model_dump(mode="json")
            elif isinstance(output, BaseModel):
                output = output.model_dump(mode="json")
            encoded = json.dumps(
                output,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, ValidationError) as exc:
            return result.model_copy(
                update={
                    "ok": False,
                    "output": None,
                    "error_code": "invalid_output",
                    "error_message": str(exc)[:2000],
                }
            )
        if len(encoded) > tool.max_output_bytes:
            return result.model_copy(
                update={
                    "ok": False,
                    "output": None,
                    "error_code": "output_too_large",
                    "error_message": (
                        f"tool output is {len(encoded)} bytes; maximum is "
                        f"{tool.max_output_bytes} bytes"
                    ),
                }
            )
        return result.model_copy(update={"output": output})

    async def _start_operation(
        self,
        call: ToolCall,
        tool: AgentTool | None,
        context: ToolContext,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        if self._execution_store is None:
            return call.operation_id, None, None
        try:
            idempotency_key = _idempotency_key(call, context)
            if call.operation_id:
                existing = await self._execution_store.get_operation(
                    operation_id=call.operation_id,
                    user_id=context.user_id,
                    workspace_id=context.workspace_id,
                    thread_id=context.thread_id,
                )
                if existing is not None:
                    expected_hash = _arguments_hash(call.arguments)
                    if existing.get("arguments_hash") not in {None, expected_hash}:
                        raise ValueError("idempotency key was reused with different tool input")
                    return call.operation_id, None, existing
            operation_id = await self._execution_store.create_operation(
                operation_id=call.operation_id,
                idempotency_key=idempotency_key,
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

    async def _claim_operation(
        self,
        operation_id: str,
        context: ToolContext,
    ) -> dict[str, Any] | None:
        if self._execution_store is None:
            return None
        if getattr(self._execution_store, "supports_atomic_claim", False) is not True:
            return None
        return await self._execution_store.claim_operation(
            operation_id=operation_id,
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            thread_id=context.thread_id,
            lease_owner=self._worker_id,
            lease_seconds=60,
        )

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
            lease_owner=None,
            lease_expires_at=None,
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


def _first_argument_value(values: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = values.get(name)
        if value is not None:
            return str(value)
    return None


def _idempotency_key(call: ToolCall, context: ToolContext) -> str:
    if call.operation_id:
        return call.operation_id
    request_id = context.request_id or context.turn_id or context.thread_id
    raw_key = f"{request_id}:{call.call_id}"
    return f"idem_{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}"


def _arguments_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
