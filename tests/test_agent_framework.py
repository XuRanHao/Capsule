from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from capsule.agent import (
    AgentRequest,
    AgentTool,
    InMemoryMemoryStore,
    RetryableToolError,
    ToolContext,
    ToolHooks,
    ToolRegistry,
    create_agent_runtime,
)
from capsule.agent.contracts import PlanDecision, ToolCall
from capsule.db.agent_memory import (
    AgentMessageRecord,
    AgentThreadRecord,
    ConversationContext,
    ConversationSyncState,
)


class EchoArgs(BaseModel):
    value: str


class EchoOutput(BaseModel):
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


class ConfirmationThenRespondPlanner:
    async def plan(self, state: object) -> PlanDecision:
        values = state if isinstance(state, dict) else {}
        messages = values.get("messages", [])
        if messages and messages[-1].get("content") == "准备操作":
            return PlanDecision(
                action="tool",
                tool_calls=[ToolCall(name="echo", arguments={"value": "queued"})],
            )
        return PlanDecision(action="respond", message="已开始新的对话轮次。")


class RecordingPlanner:
    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []

    async def plan(self, state: object) -> PlanDecision:
        values = state if isinstance(state, dict) else {}
        self.contexts.append(
            {
                "messages": list(values.get("messages", [])),
                "working_context": dict(values.get("working_context", {})),
                "memory_context": list(values.get("memory_context", [])),
            }
        )
        return PlanDecision(action="respond", message="已读取上下文。")


class FakeConversationRepository:
    def __init__(self, *, revision: int) -> None:
        now = datetime.now(UTC)
        self.current_user = AgentMessageRecord(
            message_id="current-user",
            thread_id="durable-thread",
            sequence=4,
            role="user",
            content="新问题",
            name=None,
            turn_id="turn-current",
            request_id="request-current",
            estimated_tokens=4,
            created_at=now,
        )
        self.assistant = AgentMessageRecord(
            message_id="current-assistant",
            thread_id="durable-thread",
            sequence=5,
            role="assistant",
            content="已读取上下文。",
            name=None,
            turn_id="turn-current",
            request_id="request-current",
            estimated_tokens=4,
            created_at=now,
        )
        self.context = ConversationContext(
            thread=AgentThreadRecord(
                thread_id="durable-thread",
                user_id="user-a",
                workspace_id="workspace-a",
                title="新会话",
                status="active",
                summary="此前讨论了记忆架构。",
                summary_topic="记忆架构",
                summary_covered_sequence=2,
                last_consolidated_sequence=2,
                memory_revision=revision,
                last_message_at=now,
            ),
            messages=[
                AgentMessageRecord(
                    message_id="old-user",
                    thread_id="durable-thread",
                    sequence=3,
                    role="user",
                    content="旧问题",
                    name=None,
                    turn_id="turn-old",
                    request_id="request-old",
                    estimated_tokens=4,
                    created_at=now,
                ),
                self.current_user,
            ],
        )
        self.enqueued = 0
        self.context_reads = 0
        self.sync_state = ConversationSyncState(
            last_message_sequence=3,
            memory_revision=revision,
        )

    async def enqueue_consolidation_if_needed(self, **_: object) -> None:
        self.enqueued += 1

    async def append_message(self, *, role: str, **_: object) -> AgentMessageRecord:
        return self.current_user if role == "user" else self.assistant

    async def get_context(self, **_: object) -> ConversationContext:
        self.context_reads += 1
        return self.context

    async def get_sync_state(self, **_: object) -> ConversationSyncState:
        return self.sync_state

    def prepare_next_turn(
        self,
        *,
        user_sequence: int,
        durable_last_sequence: int | None = None,
    ) -> None:
        self.current_user = replace(
            self.current_user,
            sequence=user_sequence,
            content="下一问题",
        )
        self.assistant = replace(
            self.assistant,
            sequence=user_sequence + 1,
        )
        self.sync_state = ConversationSyncState(
            last_message_sequence=(
                user_sequence - 1
                if durable_last_sequence is None
                else durable_last_sequence
            ),
            memory_revision=self.context.thread.memory_revision,
        )


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
    # First pass selects the tool from the compact catalog; its full schema is
    # disclosed before the second planning pass can execute it.
    assert planner.calls == 3
    snapshot = await runtime.state("thread-a")
    assert snapshot is not None
    assert snapshot["messages"][-1]["role"] == "assistant"
    assert "args_schema" not in snapshot["tool_catalog"][0]
    assert snapshot["tool_details"][0]["args_schema"]["title"] == "EchoArgs"


@pytest.mark.asyncio
async def test_runtime_hydrates_cold_conversation_context_from_postgres_repository() -> None:
    planner = RecordingPlanner()
    repository = FakeConversationRepository(revision=1)
    runtime = create_agent_runtime(
        planner=planner,
        conversation_repository=repository,  # type: ignore[arg-type]
    )

    result = await runtime.invoke(
        AgentRequest(
            thread_id="durable-thread",
            user_id="user-a",
            workspace_id="workspace-a",
            message="新问题",
            request_id="request-current",
        )
    )

    assert result.status == "completed"
    assert [item["content"] for item in planner.contexts[-1]["messages"]] == [
        "旧问题",
        "新问题",
    ]
    assert planner.contexts[-1]["working_context"] == {
        "conversation_summary": "此前讨论了记忆架构。",
        "conversation_topic": "记忆架构",
        "summary_covered_sequence": 2,
        "memory_revision": 1,
        "hot_mirror_through_sequence": 3,
    }


@pytest.mark.asyncio
async def test_runtime_refreshes_hot_context_after_worker_updates_summary() -> None:
    planner = RecordingPlanner()
    runtime = create_agent_runtime(planner=planner)
    request = AgentRequest(
        thread_id="durable-thread",
        user_id="user-a",
        workspace_id="workspace-a",
        message="第一次问题",
    )
    await runtime.invoke(request)

    repository = FakeConversationRepository(revision=1)
    runtime.set_conversation_repository(
        repository,  # type: ignore[arg-type]
    )
    result = await runtime.invoke(
        request.model_copy(
            update={"message": "新问题", "request_id": "request-current"}
        )
    )

    assert result.status == "completed"
    # Ordinary turns never enqueue memory work.  Consolidation is now only
    # requested by the context-budget node after an actual overflow.
    assert repository.enqueued == 0
    assert [item["content"] for item in planner.contexts[-1]["messages"]] == [
        "旧问题",
        "新问题",
    ]
    snapshot = await runtime.state("durable-thread")
    assert snapshot is not None
    assert snapshot["working_context"]["memory_revision"] == 1


@pytest.mark.asyncio
async def test_runtime_reuses_a_valid_hot_mirror_without_reloading_messages() -> None:
    planner = RecordingPlanner()
    repository = FakeConversationRepository(revision=1)
    runtime = create_agent_runtime(
        planner=planner,
        conversation_repository=repository,  # type: ignore[arg-type]
    )
    request = AgentRequest(
        thread_id="durable-thread",
        user_id="user-a",
        workspace_id="workspace-a",
        message="新问题",
        request_id="request-current",
    )

    await runtime.invoke(request)
    repository.prepare_next_turn(user_sequence=6)
    await runtime.invoke(
        request.model_copy(update={"message": "下一问题", "request_id": "request-next"})
    )

    assert repository.context_reads == 1
    snapshot = await runtime.state("durable-thread")
    assert snapshot is not None
    assert snapshot["working_context"]["hot_mirror_through_sequence"] == 7


@pytest.mark.asyncio
async def test_runtime_rebuilds_a_hot_mirror_advanced_by_another_process() -> None:
    planner = RecordingPlanner()
    repository = FakeConversationRepository(revision=1)
    runtime = create_agent_runtime(
        planner=planner,
        conversation_repository=repository,  # type: ignore[arg-type]
    )
    request = AgentRequest(
        thread_id="durable-thread",
        user_id="user-a",
        workspace_id="workspace-a",
        message="新问题",
        request_id="request-current",
    )

    await runtime.invoke(request)
    repository.prepare_next_turn(user_sequence=7, durable_last_sequence=6)
    await runtime.invoke(
        request.model_copy(update={"message": "下一问题", "request_id": "request-next"})
    )

    assert repository.context_reads == 2


@pytest.mark.asyncio
async def test_runtime_uses_memory_reader_attached_after_graph_creation() -> None:
    planner = RecordingPlanner()

    class MemoryReader:
        async def load(self, **_: object) -> list[dict[str, object]]:
            return [{"scope": "workspace", "text": "使用中文"}]

        async def save(self, **_: object) -> None:
            return None

    runtime = create_agent_runtime(planner=planner)
    runtime.set_memory_store(MemoryReader())  # type: ignore[arg-type]

    result = await runtime.invoke(
        AgentRequest(
            thread_id="thread-memory-reader",
            user_id="user-a",
            workspace_id="workspace-a",
            message="继续讨论",
        )
    )

    assert result.status == "completed"
    assert planner.contexts[-1]["memory_context"] == [
        {"scope": "workspace", "text": "使用中文"}
    ]


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
    assert len({item["turn_id"] for item in resumed.tool_history}) == 1


@pytest.mark.asyncio
async def test_confirmation_cancel_marks_pending_tool_calls_cancelled() -> None:
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
        thread_id="thread-cancel-queue",
        user_id="user-a",
        workspace_id="workspace-a",
        message="执行写操作",
    )
    pending = await runtime.invoke(request)
    assert pending.status == "awaiting_confirmation"

    cancelled = await runtime.invoke(
        request.model_copy(update={"message": "取消", "confirmation": False})
    )
    assert cancelled.status == "cancelled"
    snapshot = await runtime.state(request.thread_id)
    assert snapshot is not None
    assert snapshot["pending_tool_calls"][0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_new_round_clears_old_pending_tool_queue() -> None:
    runtime = create_agent_runtime(
        planner=ConfirmationThenRespondPlanner(),
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
        thread_id="thread-new-round-queue",
        user_id="user-a",
        workspace_id="workspace-a",
        message="准备操作",
    )
    pending = await runtime.invoke(request)
    assert pending.status == "awaiting_confirmation"
    snapshot = await runtime.state(request.thread_id)
    assert snapshot is not None
    assert snapshot["pending_tool_calls"]

    result = await runtime.invoke(request.model_copy(update={"message": "新的请求"}))
    assert result.status == "completed"
    snapshot = await runtime.state(request.thread_id)
    assert snapshot is not None
    assert snapshot["pending_tool_calls"] == []


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
    async def slow_handler(args: EchoArgs, context: ToolContext) -> None:
        del args, context
        await asyncio.sleep(0.05)

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
                handler=slow_handler,
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
    assert timed_out.error_code == "timeout"


@pytest.mark.asyncio
async def test_retryable_tool_error_uses_backoff_and_succeeds() -> None:
    attempts = 0

    async def handler(args: EchoArgs, context: ToolContext) -> str:
        nonlocal attempts
        del context
        attempts += 1
        if attempts < 3:
            raise RetryableToolError("temporary dependency failure")
        return args.value

    registry = ToolRegistry(
        [
            AgentTool(
                name="retryable",
                description="retryable test tool",
                args_schema=EchoArgs,
                handler=handler,
                max_attempts=3,
                retry_backoff_seconds=0,
                retry_jitter_seconds=0,
            )
        ]
    )
    result = await registry.execute(
        ToolCall(name="retryable", arguments={"value": "ok"}),
        context=ToolContext(
            user_id="user-a",
            workspace_id="workspace-a",
            thread_id="thread-retry",
            graph_id=None,
            state={},
        ),
    )

    assert result.ok is True
    assert result.attempts == 3
    assert attempts == 3


@pytest.mark.asyncio
async def test_non_retryable_tool_error_stops_without_repeating() -> None:
    attempts = 0

    def handler(args: EchoArgs, context: ToolContext) -> str:
        nonlocal attempts
        del args, context
        attempts += 1
        raise ValueError("invalid business state")

    registry = ToolRegistry(
        [
            AgentTool(
                name="non-retryable",
                description="non-retryable test tool",
                args_schema=EchoArgs,
                handler=handler,
                max_attempts=3,
            )
        ]
    )
    result = await registry.execute(
        ToolCall(name="non-retryable", arguments={"value": "x"}),
        context=ToolContext(
            user_id="user-a",
            workspace_id="workspace-a",
            thread_id="thread-retry-no-repeat",
            graph_id=None,
            state={},
        ),
    )

    assert result.ok is False
    assert result.error_code == "execution_failed"
    assert result.attempts == 1
    assert attempts == 1


@pytest.mark.asyncio
async def test_retryable_failure_reports_retry_exhausted() -> None:
    async def handler(args: EchoArgs, context: ToolContext) -> str:
        del args, context
        raise ConnectionError("temporary network failure")

    registry = ToolRegistry(
        [
            AgentTool(
                name="exhausted",
                description="retry exhaustion test tool",
                args_schema=EchoArgs,
                handler=handler,
                max_attempts=2,
                retry_backoff_seconds=0,
                retry_jitter_seconds=0,
            )
        ]
    )
    result = await registry.execute(
        ToolCall(name="exhausted", arguments={"value": "x"}),
        context=ToolContext(
            user_id="user-a",
            workspace_id="workspace-a",
            thread_id="thread-retry-exhausted",
            graph_id=None,
            state={},
        ),
    )

    assert result.ok is False
    assert result.error_code == "retry_exhausted"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_running_tool_can_stop_at_a_cooperative_cancellation_point() -> None:
    started = asyncio.Event()

    async def cancellable_handler(args: EchoArgs, context: ToolContext) -> str:
        del args
        started.set()
        while True:
            context.raise_if_cancelled()
            await asyncio.sleep(0.001)

    registry = ToolRegistry(
        [
            AgentTool(
                name="cancellable",
                description="cooperative cancellation test",
                args_schema=EchoArgs,
                handler=cancellable_handler,
            )
        ]
    )
    call = ToolCall(name="cancellable", arguments={"value": "x"})
    task = asyncio.create_task(
        registry.execute(
            call,
            context=ToolContext(
                user_id="user-a",
                workspace_id="workspace-a",
                thread_id="thread-cancel-running",
                graph_id=None,
                state={},
            ),
        )
    )
    await started.wait()
    assert registry.request_cancel(call.call_id) is True

    result = await task
    assert result.ok is False
    assert result.error_code == "cancelled"


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


@pytest.mark.asyncio
async def test_tool_output_schema_and_size_are_enforced() -> None:
    registry = ToolRegistry(
        [
            AgentTool(
                name="typed",
                description="typed output",
                args_schema=EchoArgs,
                output_schema=EchoOutput,
                handler=lambda args, context: {"value": args.value},
            ),
            AgentTool(
                name="wrong",
                description="wrong output",
                args_schema=EchoArgs,
                output_schema=EchoOutput,
                handler=lambda args, context: {"other": args.value},
            ),
            AgentTool(
                name="large",
                description="bounded output",
                args_schema=EchoArgs,
                max_output_bytes=8,
                handler=lambda args, context: {"value": args.value},
            ),
        ]
    )
    typed = await registry.execute(
        ToolCall(name="typed", arguments={"value": "ok"}),
        context=None,  # type: ignore[arg-type]
    )
    wrong = await registry.execute(
        ToolCall(name="wrong", arguments={"value": "ok"}),
        context=None,  # type: ignore[arg-type]
    )
    large = await registry.execute(
        ToolCall(name="large", arguments={"value": "too large"}),
        context=None,  # type: ignore[arg-type]
    )

    assert typed.ok is True
    assert typed.output == {"value": "ok"}
    assert wrong.error_code == "invalid_output"
    assert large.error_code == "output_too_large"


@pytest.mark.asyncio
async def test_exclusive_tools_lock_the_declared_resource_but_parallel_tools_do_not() -> None:
    active = 0
    max_active = 0

    async def handler(args: EchoArgs, context: ToolContext) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return args.value

    context = ToolContext(
        user_id="user-a",
        workspace_id="workspace-a",
        thread_id="thread-a",
        graph_id="graph-a",
        state={},
    )
    exclusive = ToolRegistry(
        [
            AgentTool(
                name="write",
                description="exclusive graph write",
                args_schema=EchoArgs,
                handler=handler,
                concurrency_mode="exclusive",
                lock_scope="graph",
            )
        ]
    )
    await asyncio.gather(
        exclusive.execute(ToolCall(name="write", arguments={"value": "a"}), context=context),
        exclusive.execute(ToolCall(name="write", arguments={"value": "b"}), context=context),
    )
    assert max_active == 1

    active = 0
    max_active = 0
    parallel = ToolRegistry(
        [
            AgentTool(
                name="read",
                description="parallel graph read",
                args_schema=EchoArgs,
                handler=handler,
                concurrency_mode="parallel",
            )
        ]
    )
    await asyncio.gather(
        parallel.execute(ToolCall(name="read", arguments={"value": "a"}), context=context),
        parallel.execute(ToolCall(name="read", arguments={"value": "b"}), context=context),
    )
    assert max_active == 2


@pytest.mark.asyncio
async def test_injected_lock_provider_serializes_two_registries() -> None:
    class SharedLock:
        def __init__(self, lock: asyncio.Lock) -> None:
            self._lock = lock

        async def __aenter__(self) -> None:
            await self._lock.acquire()

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            self._lock.release()

    class SharedLockProvider:
        def __init__(self) -> None:
            self._locks: dict[str, asyncio.Lock] = {}

        def lock(
            self,
            key: str,
            *,
            lease_seconds: float,
            wait_seconds: float,
        ) -> SharedLock:
            del lease_seconds, wait_seconds
            return SharedLock(self._locks.setdefault(key, asyncio.Lock()))

    active = 0
    max_active = 0

    async def handler(args: EchoArgs, context: ToolContext) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return args.value

    provider = SharedLockProvider()
    context = ToolContext(
        user_id="user-a",
        workspace_id="workspace-a",
        thread_id="thread-a",
        graph_id="graph-a",
        state={},
    )
    registries = [
        ToolRegistry(
            [
                AgentTool(
                    name="write",
                    description="exclusive graph write",
                    args_schema=EchoArgs,
                    handler=handler,
                    concurrency_mode="exclusive",
                    lock_scope="graph",
                )
            ],
            lock_provider=provider,
        )
        for _ in range(2)
    ]
    await asyncio.gather(
        registries[0].execute(ToolCall(name="write", arguments={"value": "a"}), context=context),
        registries[1].execute(ToolCall(name="write", arguments={"value": "b"}), context=context),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_registry_claims_idempotent_operation_before_handler() -> None:
    execution_store = AsyncMock()
    execution_store.supports_atomic_claim = True
    execution_store.create_operation.return_value = "op_claim"
    execution_store.claim_operation.return_value = {
        "claimed": True,
        "operation_id": "op_claim",
    }
    handler = AsyncMock(return_value={"ok": True})
    registry = ToolRegistry(
        [
            AgentTool(
                name="echo",
                description="claimed operation",
                args_schema=EchoArgs,
                handler=handler,
            )
        ],
        execution_store=execution_store,
    )
    context = ToolContext(
        user_id="user-a",
        workspace_id="workspace-a",
        thread_id="thread-a",
        graph_id=None,
        state={},
        request_id="request-a",
        turn_id="turn-a",
    )

    result = await registry.execute(
        ToolCall(name="echo", arguments={"value": "x"}),
        context=context,
    )

    assert result.ok is True
    execution_store.create_operation.assert_awaited_once()
    execution_store.create_operation.assert_awaited_once_with(
        operation_id=None,
        idempotency_key=(
            "idem_"
            + hashlib.sha256(
                f"request-a:{result.call_id}".encode()
            ).hexdigest()
        ),
        call_id=result.call_id,
        thread_id="thread-a",
        user_id="user-a",
        workspace_id="workspace-a",
        graph_id=None,
        tool_name="echo",
        required_permission=None,
        arguments={"value": "x"},
        confirmation_status="not_required",
        execution_status="created",
    )
    execution_store.claim_operation.assert_awaited_once()
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_input_validator_runs_after_schema_validation() -> None:
    handler = AsyncMock(return_value="should not run")

    async def validate(args: EchoArgs, context: ToolContext) -> str | None:
        del context
        return "value cannot be empty" if not args.value.strip() else None

    registry = ToolRegistry(
        [
            AgentTool(
                name="validated",
                description="validated input",
                args_schema=EchoArgs,
                handler=handler,
                validate_input=validate,
            )
        ]
    )

    result = await registry.execute(
        ToolCall(name="validated", arguments={"value": "   "}),
        context=ToolContext(
            user_id="user-a",
            workspace_id="workspace-a",
            thread_id="thread-a",
            graph_id=None,
            state={},
        ),
    )

    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert result.error_message == "value cannot be empty"
    handler.assert_not_awaited()
