import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from capsule.agent.contracts import AgentRequest, PlanDecision
from capsule.agent.model_planner import AgentPlanningError, ModelAgentPlanner
from capsule.agent.runtime import AgentRuntime


class RecordingStructuredModel:
    def __init__(self, response: PlanDecision) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def generate_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[PlanDecision],
        schema_name: str,
        max_output_tokens: int,
        model: str | None = None,
    ) -> PlanDecision:
        self.requests.append(
            {
                "messages": list(messages),
                "output_type": output_type,
                "schema_name": schema_name,
                "max_output_tokens": max_output_tokens,
                "model": model,
            }
        )
        return self.response


class FailingStructuredModel:
    async def generate_structured(self, **_: object) -> PlanDecision:
        raise ConnectionError("model gateway is unavailable")


@pytest.mark.asyncio
async def test_model_planner_projects_only_governed_planner_context() -> None:
    model = RecordingStructuredModel(PlanDecision(action="respond", message="已处理。"))
    planner = ModelAgentPlanner(
        model=model,
        model_name="agent-model-test",
        max_output_tokens=700,
    )

    decision = await planner.plan(
        {
            "messages": [{"role": "user", "content": "整理这个工作区"}],
            "working_context": {
                "conversation_summary": "此前讨论了素材归类。",
                "conversation_topic": "素材整理",
                "memory_revision": 2,
            },
            "memory_context": [{"scope": "workspace", "text": "使用中文"}],
            "tool_history": [{"name": "search", "ok": True, "output": {"id": "a"}}],
            "tool_catalog": [{"name": "search", "description": "搜索素材"}],
            "tool_details": [{"name": "search", "args_schema": {"type": "object"}}],
            "deferred_tool_names": ["update_asset"],
            "granted_permissions": ["asset:read"],
            # Checkpoint-control state must not be offered as model authority.
            "pending_action": {"action": "tool"},
            "turn_id": "turn-secret",
        }
    )

    assert decision.message == "已处理。"
    request = model.requests[0]
    assert request["output_type"] is PlanDecision
    assert request["schema_name"] == "capsule_agent_plan"
    assert request["max_output_tokens"] == 700
    assert request["model"] == "agent-model-test"
    prompt = request["messages"]
    assert isinstance(prompt, list)
    payload = json.loads(str(prompt[1]["content"]))
    assert payload == {
        "conversation": {
            "summary": "此前讨论了素材归类。",
            "topic": "素材整理",
            "messages": [{"role": "user", "content": "整理这个工作区"}],
        },
        "recalled_memories": [{"scope": "workspace", "text": "使用中文"}],
        "recent_tool_results": [{"name": "search", "ok": True, "output": {"id": "a"}}],
        "tools": {
            "catalog": [{"name": "search", "description": "搜索素材"}],
            "detailed_definitions": [
                {"name": "search", "args_schema": {"type": "object"}}
            ],
            "deferred_names": ["update_asset"],
        },
        "granted_permissions": ["asset:read"],
    }


@pytest.mark.asyncio
async def test_model_planner_wraps_remote_failures_as_recoverable_planning_errors() -> None:
    planner = ModelAgentPlanner(
        model=FailingStructuredModel(),  # type: ignore[arg-type]
        model_name="agent-model-test",
        max_output_tokens=700,
    )

    with pytest.raises(AgentPlanningError):
        await planner.plan({})


@pytest.mark.asyncio
async def test_runtime_rebuilds_with_model_planner_and_returns_a_safe_model_failure() -> None:
    runtime = AgentRuntime()
    runtime.set_planner(
        ModelAgentPlanner(
            model=FailingStructuredModel(),  # type: ignore[arg-type]
            model_name="agent-model-test",
            max_output_tokens=700,
        )
    )

    result = await runtime.invoke(
        AgentRequest(
            thread_id="model-planner-failure",
            user_id="user-a",
            workspace_id="workspace-a",
            message="继续讨论",
        )
    )

    assert result.status == "failed"
    assert result.message == "对话规划服务暂不可用，请稍后重试。"
