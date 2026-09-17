"""Model-backed planner that converts governed Agent state into PlanDecision."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from capsule.agent.contracts import PlanDecision

logger = logging.getLogger(__name__)


class AgentPlanningError(RuntimeError):
    """A recoverable remote-model failure while creating an Agent plan."""


class StructuredPlanModel(Protocol):
    """Strict structured-output capability required by the conversation planner."""

    async def generate_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[PlanDecision],
        schema_name: str,
        max_output_tokens: int,
        model: str | None = None,
    ) -> PlanDecision: ...


class ModelAgentPlanner:
    """Plan one governed conversational turn through a strict JSON model call.

    The graph supplies only context that has already passed context-budget
    governance. This adapter does not grant permissions or execute tools; its
    output is still checked by the graph and ToolRegistry.
    """

    def __init__(
        self,
        *,
        model: StructuredPlanModel,
        model_name: str,
        max_output_tokens: int,
    ) -> None:
        if not model_name.strip():
            raise ValueError("agent planner model must not be blank")
        if max_output_tokens < 1:
            raise ValueError("agent planner max output tokens must be positive")
        self._model = model
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    async def plan(self, state: Mapping[str, object]) -> PlanDecision:
        try:
            return await self._model.generate_structured(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            _planner_payload(state),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                ],
                output_type=PlanDecision,
                schema_name="capsule_agent_plan",
                max_output_tokens=self._max_output_tokens,
                model=self._model_name,
            )
        except Exception as exc:
            logger.warning(
                "agent planner model request failed error_type=%s",
                type(exc).__name__,
                exc_info=True,
            )
            raise AgentPlanningError("agent planner model is unavailable") from exc


def _planner_payload(state: Mapping[str, object]) -> dict[str, object]:
    """Project only planner-visible state; do not leak checkpoint control fields."""

    working_context = _as_mapping(state.get("working_context"))
    return {
        "conversation": {
            "summary": working_context.get("conversation_summary"),
            "topic": working_context.get("conversation_topic"),
            "messages": _as_mappings(state.get("messages")),
        },
        "recalled_memories": _as_mappings(state.get("memory_context")),
        "recent_tool_results": _as_mappings(state.get("tool_history")),
        "tools": {
            "catalog": _as_mappings(state.get("tool_catalog")),
            "detailed_definitions": _as_mappings(state.get("tool_details")),
            "deferred_names": _as_strings(state.get("deferred_tool_names")),
        },
        "granted_permissions": _as_strings(state.get("granted_permissions")),
    }


def _as_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _as_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


_SYSTEM_PROMPT = """你是 Capsule 的会话规划器。基于输入状态，
输出完全符合提供 JSON Schema 的 PlanDecision。
不要输出解释、Markdown 或 Schema 以外字段。

目标是决定下一步，而不是执行工具：
1. 可以直接回答时，使用 respond 并在 message 中给出简洁、可靠的中文回复。
2. 需要工具且完整定义尚未出现在 tools.detailed_definitions 时，只能使用 select_tools。
   填写 catalog 中的名称，不能填写 tool_calls。
3. 仅当完整定义已经出现时，才可使用 tool 或 confirm。调用名必须来自 detailed_definitions，
   参数必须严格匹配 args_schema，绝不编造工具或字段。
4. 工具标记 requires_confirmation=true，或动作会造成写入、外发、删除、权限变更等风险时，
   优先使用 confirm 并提供 confirmation_message。用户确认由服务端恢复执行，
   不由你假定已经发生。
5. recent_tool_results 是已发生动作的事实。根据成功或失败结果继续处理或回复，
   不能声称尚未成功的工具已经完成。
6. recalled_memories 是辅助上下文；当前用户消息、服务端权限和工具 Schema 优先。
   不要写入、要求确认或臆造长期记忆。
7. 若 detailed_definitions 被延后，请重新选择所需工具；不要绕过披露流程。"""
