from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from capsule.agent import (
    AgentRequest,
    AgentTool,
    InMemoryMemoryStore,
    ToolHooks,
    ToolRegistry,
    create_agent_runtime,
)
from capsule.agent.contracts import PlanDecision, ToolCall


class EchoArgs(BaseModel):
    value: str


class SequencePlanner:
    def __init__(self, *, confirmation: bool = False) -> None:
        self.calls = 0
        self.confirmation = confirmation

    async def plan(self, state: object) -> PlanDecision:
        self.calls += 1
        values = state if isinstance(state, dict) else {}
        if self.confirmation and not values.get("tool_history"):
            return PlanDecision(
                action="confirm",
                confirmation_message="确认执行素材操作？",
                tool_calls=[ToolCall(name="echo", arguments={"value": "confirmed"})],
            )
        if not values.get("tool_history"):
            return PlanDecision(
                action="tool",
                tool_calls=[ToolCall(name="echo", arguments={"value": "hello"})],
            )
        return PlanDecision(action="respond", message="已完成。")


class DirectWritePlanner:
    async def plan(self, state: object) -> PlanDecision:
        values = state if isinstance(state, dict) else {}
        if not values.get("tool_history"):
            return PlanDecision(
                action="tool",
                tool_calls=[ToolCall(name="echo", arguments={"value": "guarded"})],
            )
        return PlanDecision(action="respond", message="已完成。")


class OneToolPerOutputPlanner:
    async def plan(self, state: object) -> PlanDecision:
        values = state if isinstance(state, dict) else {}
        messages = values.get("messages", [])
        if messages and messages[-1].get("role") == "user":
            return PlanDecision(
                action="tool",
                tool_calls=[ToolCall(name="echo", arguments={"value": "round"})],
            )
        return PlanDecision(action="respond", message="本轮完成。")


@pytest.mark.asyncio
async def test_runtime_runs_tool_loop_and_persists_thread_state() -> None:
    planner = SequencePlanner()
    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="return the supplied value",
                args_schema=EchoArgs,
                handler=lambda args, context: {
                    "value": args.value,
                    "workspace": context.workspace_id,
                },
            )
        ]
    )
    runtime = create_agent_runtime(planner=planner, tools=registry)

    result = await runtime.invoke(
        AgentRequest(
            thread_id="thread-a",
            user_id="user-a",
            workspace_id="workspace-a",
            message="执行一次测试工具",
        )
    )

    assert result.status == "completed"
    assert result.message == "已完成。"
    assert result.tool_history[0]["output"]["workspace"] == "workspace-a"
    assert planner.calls == 2
    snapshot = await runtime.state("thread-a")
    assert snapshot is not None
    assert snapshot["messages"][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_tool_history_keeps_only_the_latest_two_agent_output_rounds() -> None:
    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="return the supplied value",
                args_schema=EchoArgs,
                handler=lambda args, context: args.value,
            )
        ]
    )
    runtime = create_agent_runtime(
        planner=OneToolPerOutputPlanner(),
        tools=registry,
    )

    for index in range(3):
        result = await runtime.invoke(
            AgentRequest(
                thread_id="thread-round-retention",
                user_id="user-a",
                workspace_id="workspace-a",
                message=f"第 {index + 1} 轮",
            )
        )
        assert result.status == "completed"

    assert len(result.tool_history) == 2
    turn_ids = [item["turn_id"] for item in result.tool_history]
    assert len(set(turn_ids)) == 2
    snapshot = await runtime.state("thread-round-retention")
    assert snapshot is not None
    assert len(snapshot["tool_history"]) == 2


@pytest.mark.asyncio
async def test_write_tool_requires_confirmation_and_can_resume() -> None:
    planner = SequencePlanner(confirmation=True)
    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="confirmed test operation",
                args_schema=EchoArgs,
                handler=lambda args, context: args.value,
                requires_confirmation=True,
            )
        ]
    )
    runtime = create_agent_runtime(planner=planner, tools=registry)
    request = AgentRequest(
        thread_id="thread-confirm",
        user_id="user-a",
        workspace_id="workspace-a",
        message="准备执行",
    )

    pending = await runtime.invoke(request)
    assert pending.status == "awaiting_confirmation"
    assert pending.pending_action is not None

    resumed = await runtime.invoke(
        request.model_copy(update={"message": "确认", "confirmation": True})
    )
    assert resumed.status == "completed"
    assert resumed.tool_history[0]["ok"] is True


@pytest.mark.asyncio
async def test_registry_confirmation_guard_interrupts_direct_write_plan() -> None:
    runtime = create_agent_runtime(
        planner=DirectWritePlanner(),
        tools=ToolRegistry(
            [
                AgentTool(
                    name="echo",
                    description="guarded test operation",
                    args_schema=EchoArgs,
                    handler=lambda args, context: args.value,
                    requires_confirmation=True,
                )
            ]
        ),
    )
    request = AgentRequest(
        thread_id="thread-guard",
        user_id="user-a",
        workspace_id="workspace-a",
        message="执行写操作",
    )
    pending = await runtime.invoke(request)
    assert pending.status == "awaiting_confirmation"
    resumed = await runtime.invoke(
        request.model_copy(update={"message": "确认", "confirmation": True})
    )
    assert resumed.status == "completed"
    assert resumed.tool_history[-1]["ok"] is True


@pytest.mark.asyncio
async def test_runtime_loads_user_permissions_for_each_request() -> None:
    calls: list[tuple[str, str]] = []

    async def load_permissions(*, user_id: str, workspace_id: str) -> list[str]:
        calls.append((user_id, workspace_id))
        return ["graph:write"]

    runtime = create_agent_runtime(
        planner=DirectWritePlanner(),
        permission_loader=load_permissions,
        tools=ToolRegistry(
            [
                AgentTool(
                    name="echo",
                    description="permission-scoped operation",
                    args_schema=EchoArgs,
                    handler=lambda args, context: args.value,
                    required_permission="graph:write",
                )
            ]
        ),
    )

    result = await runtime.invoke(
        AgentRequest(
            thread_id="thread-permission",
            user_id="user-a",
            workspace_id="workspace-a",
            message="执行普通写操作",
        )
    )

    assert result.status == "completed"
    assert result.tool_history[0]["ok"] is True
    assert calls == [("user-a", "workspace-a")]


@pytest.mark.asyncio
async def test_confirmation_keeps_operation_id_for_resume() -> None:
    execution_store = AsyncMock()
    execution_store.create_operation.return_value = "op_confirm"
    execution_store.get_operation.return_value = {"execution_status": "awaiting_confirmation"}
    handler = AsyncMock(return_value="done")
    runtime = create_agent_runtime(
        planner=DirectWritePlanner(),
        tools=ToolRegistry(
            [
                AgentTool(
                    name="echo",
                    description="confirmed operation",
                    args_schema=EchoArgs,
                    handler=handler,
                    requires_confirmation=True,
                )
            ],
            execution_store=execution_store,
        ),
    )
    request = AgentRequest(
        thread_id="thread-operation-id",
        user_id="user-a",
        workspace_id="workspace-a",
        message="执行操作",
    )

    pending = await runtime.invoke(request)
    assert pending.status == "awaiting_confirmation"
    assert pending.pending_action is not None
    assert pending.pending_action["tool_calls"][0]["operation_id"] == "op_confirm"

    resumed = await runtime.invoke(
        request.model_copy(update={"message": "确认", "confirmation": True})
    )
    assert resumed.status == "completed"
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_is_scoped_by_user_and_workspace() -> None:
    memory = InMemoryMemoryStore()
    await memory.save(
        user_id="user-a",
        workspace_id="workspace-a",
        writes=[],
    )
    assert await memory.load(user_id="user-a", workspace_id="workspace-a", query="") == []
    assert await memory.load(user_id="user-b", workspace_id="workspace-a", query="") == []


@pytest.mark.asyncio
async def test_tool_schema_and_timeout_fail_as_structured_results() -> None:
    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="echo",
                args_schema=EchoArgs,
                handler=lambda args, context: args.value,
            ),
            AgentTool(
                name="slow",
                description="slow",
                args_schema=EchoArgs,
                handler=lambda args, context: asyncio.sleep(0.05),
                timeout_seconds=0.001,
            ),
        ]
    )
    invalid = await registry.execute(
        ToolCall(name="echo", arguments={}),
        context=None,  # type: ignore[arg-type]
    )
    timed_out = await registry.execute(
        ToolCall(name="slow", arguments={"value": "x"}),
        context=None,  # type: ignore[arg-type]
    )
    assert invalid.error_code == "invalid_arguments"
    assert timed_out.error_code == "execution_failed"


@pytest.mark.asyncio
async def test_tool_hooks_run_in_one_registry_pipeline() -> None:
    events: list[str] = []

    def before(call: ToolCall, tool: AgentTool, context: object) -> None:
        del call, tool, context
        events.append("before")

    def handler(args: EchoArgs, context: object) -> str:
        del args, context
        events.append("handler")
        return "raw"

    def after(result: object, call: ToolCall, tool: AgentTool, context: object):
        del call, tool, context
        events.append("after")
        return result.model_copy(update={"output": {"value": result.output}})

    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="echo",
                args_schema=EchoArgs,
                handler=handler,
            )
        ],
        hooks=ToolHooks(before=(before,), after=(after,)),
    )

    result = await registry.execute(
        ToolCall(name="echo", arguments={"value": "x"}),
        context=None,  # type: ignore[arg-type]
    )

    assert result.ok is True
    assert result.output == {"value": "raw"}
    assert events == ["before", "handler", "after"]
