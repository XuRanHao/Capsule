import os
from uuid import uuid4

import pytest
from pydantic import BaseModel

from capsule.agent.contracts import AgentRequest, PlanDecision, ToolCall
from capsule.agent.runtime import create_agent_runtime
from capsule.agent.tools import AgentTool, ToolContext, ToolRegistry
from capsule.api.app import _open_agent_checkpointer
from capsule.config import Settings


class _EchoArgs(BaseModel):
    value: str


class _ConfirmationPlanner:
    async def plan(self, state: dict[str, object]) -> PlanDecision:
        if not state.get("tool_history"):
            return PlanDecision(
                action="confirm",
                confirmation_message="确认执行？",
                tool_calls=[ToolCall(name="checkpoint_echo", arguments={"value": "ok"})],
            )
        return PlanDecision(action="respond", message="已恢复并完成。")


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_POSTGRES_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_POSTGRES_INTEGRATION=1 to exercise LangGraph checkpoints",
)
async def test_postgres_checkpoint_restores_a_pending_confirmation_after_runtime_rebuild() -> None:
    thread_id = f"checkpoint-{uuid4().hex}"
    settings = Settings()
    registry = ToolRegistry(
        [
            AgentTool(
                name="checkpoint_echo",
                description="durable checkpoint test tool",
                args_schema=_EchoArgs,
                handler=lambda args, context: _echo(args, context),
                requires_confirmation=True,
            )
        ]
    )
    request = AgentRequest(
        thread_id=thread_id,
        user_id="checkpoint-user",
        workspace_id="checkpoint-workspace",
        message="执行需要确认的操作",
        request_id=f"request-{uuid4().hex}",
    )

    first_runtime = create_agent_runtime(
        planner=_ConfirmationPlanner(),
        tools=registry,
    )
    checkpointer, checkpoint_context = await _open_agent_checkpointer(
        settings=settings,
        runtime=first_runtime,
    )
    try:
        pending = await first_runtime.invoke(request)
        assert pending.status == "awaiting_confirmation"

        resumed_runtime = create_agent_runtime(
            planner=_ConfirmationPlanner(),
            tools=registry,
            checkpointer=checkpointer,
        )
        completed = await resumed_runtime.invoke(
            request.model_copy(
                update={
                    "message": "确认",
                    "confirmation": True,
                    "request_id": f"resume-{uuid4().hex}",
                }
            )
        )
        assert completed.status == "completed"
        assert completed.message == "已恢复并完成。"
        await resumed_runtime.discard_state(thread_id)
        assert await resumed_runtime.state(thread_id) is None
    finally:
        await checkpointer.adelete_thread(thread_id)
        await checkpoint_context.__aexit__(None, None, None)


def _echo(args: _EchoArgs, context: ToolContext) -> str:
    del context
    return args.value
