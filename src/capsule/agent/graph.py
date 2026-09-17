"""LangGraph state machine used by the conversational Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from capsule.agent.context_budget import ContextBudgetController
from capsule.agent.contracts import PlanDecision, ToolCall
from capsule.agent.memory import AgentMemoryStore
from capsule.agent.model_planner import AgentPlanningError
from capsule.agent.state import AgentState
from capsule.agent.tools import ToolContext, ToolRegistry


class AgentPlanner(Protocol):
    async def plan(self, state: Mapping[str, object]) -> PlanDecision: ...


class ReadyPlanner:
    """Safe default until a model-backed planner is supplied by the application."""

    async def plan(self, state: Mapping[str, object]) -> PlanDecision:
        del state
        return PlanDecision(
            action="respond",
            message="Agent 框架已就绪，等待接入作品叙事规划器。",
        )


def _retain_recent_tool_rounds(
    history: list[dict[str, object]], *, max_rounds: int = 2
) -> list[dict[str, object]]:
    """Keep tool results from at most the latest complete Agent outputs.

    Every result in the current Agent flow carries a turn ID. Entries without
    one are not part of the current history contract and are discarded.
    """

    if max_rounds <= 0:
        return []

    round_keys: list[str] = []
    keyed_history: list[tuple[dict[str, object], str]] = []
    for item in history:
        turn_id = item.get("turn_id")
        if not turn_id:
            continue
        key = str(turn_id)
        keyed_history.append((item, key))

    for _, key in reversed(keyed_history):
        if key not in round_keys:
            round_keys.append(key)
            if len(round_keys) == max_rounds:
                break

    retained = set(round_keys)
    return [item for item, key in keyed_history if key in retained]


def _queued_tool_call(call: ToolCall, *, turn_id: str) -> dict[str, object]:
    """Serialize one planned call into the server-side execution queue."""

    return {
        "call_id": call.call_id,
        "operation_id": call.operation_id,
        "name": call.name,
        "turn_id": turn_id,
        "status": "queued",
    }


def _cancel_queued_calls(
    calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Cancel work that has not started while preserving terminal history."""

    cancelled: list[dict[str, object]] = []
    for call in calls:
        item = dict(call)
        if item.get("status") in {
            "queued",
            "awaiting_confirmation",
        }:
            item["status"] = "cancelled"
        cancelled.append(item)
    return cancelled


def build_agent_graph(
    *,
    planner: AgentPlanner,
    tools: ToolRegistry,
    memory: AgentMemoryStore,
    checkpointer: BaseCheckpointSaver[Any],
    context_budget: ContextBudgetController | None = None,
) -> Any:
    """Compile the resumable Agent graph with injected application dependencies."""

    async def prepare(state: AgentState) -> dict[str, object]:
        messages = list(state.get("messages", []))
        input_message = state.get("input_message")
        if input_message:
            messages.append({"role": "user", "content": input_message})

        current_step = state.get("step_count", 0)
        next_step = current_step + 1 if isinstance(current_step, int) else 1
        updates: dict[str, object] = {
            "messages": messages,
            "input_message": None,
            "confirmation_response": None,
            "response": None,
            "error": None,
            "step_count": next_step,
            "status": "planning",
            "start_new_turn": False,
            "tool_catalog": tools.catalog(),
        }
        context = ToolContext(
            user_id=state["user_id"],
            workspace_id=state["workspace_id"],
            thread_id=state["thread_id"],
            graph_id=state.get("graph_id"),
            state=state,
            granted_permissions=frozenset(state.get("granted_permissions", [])),
            turn_id=state.get("turn_id", ""),
            request_id=state.get("request_id", ""),
        )
        if state.get("start_new_turn"):
            # A new user round cannot inherit calls left queued by an older
            # round. Confirmation resumes keep the same turn and skip this.
            await _cancel_pending_operations(state, context)
            updates["pending_tool_calls"] = []
            updates["pending_action"] = None
            updates["approved_action"] = None
            updates["tool_details"] = []
            updates["deferred_tool_names"] = []
            updates["tool_disclosure_attempts"] = 0
        pending = state.get("pending_action")
        confirmation = state.get("confirmation_response")
        if state.get("cancel_requested"):
            await _cancel_pending_operations(state, context)
            updates["pending_tool_calls"] = _cancel_queued_calls(
                list(state.get("pending_tool_calls", []))
            )
            updates["pending_action"] = None
            updates["approved_action"] = None
            updates["response"] = "已主动终止当前待执行操作。"
            updates["status"] = "cancelled"
        elif pending is not None and confirmation is not None:
            if confirmation:
                approved = dict(pending)
                approved["action"] = "tool"
                updates["approved_action"] = approved
                updates["pending_action"] = None
            else:
                await _cancel_pending_operations(state, context)
                updates["pending_tool_calls"] = _cancel_queued_calls(
                    list(state.get("pending_tool_calls", []))
                )
                updates["pending_action"] = None
                updates["approved_action"] = None
                updates["response"] = "已取消待执行操作。"
                updates["status"] = "cancelled"
        max_steps = state.get("max_steps", 8)
        if next_step > (max_steps if isinstance(max_steps, int) else 8):
            updates["response"] = "已达到本次会话的最大执行步数。"
            updates["status"] = "max_steps"
        return updates

    async def _cancel_pending_operations(
        state: AgentState, context: ToolContext
    ) -> None:
        for item in state.get("pending_tool_calls", []):
            if item.get("status") == "running":
                tools.request_cancel(str(item.get("call_id", "")))
            elif item.get("status") in {"queued", "awaiting_confirmation"}:
                operation_id = item.get("operation_id")
                if operation_id:
                    await tools.cancel_operation(str(operation_id), context)

    async def load_context(state: AgentState) -> dict[str, object]:
        request_id = state.get("request_id")
        if request_id and state.get("memory_context_request_id") == request_id:
            # Do not let a background Worker replace recalled long/global
            # memories half-way through a tool loop in this logical request.
            return {}
        messages = state.get("messages", [])
        query = str(messages[-1].get("content", "")) if messages else ""
        context = await memory.load(
            user_id=state["user_id"],
            workspace_id=state["workspace_id"],
            query=query,
        )
        return {
            "memory_context": context,
            "memory_context_request_id": request_id,
        }

    async def manage_context(state: AgentState) -> dict[str, object]:
        if state.get("status") == "failed":
            return {}
        if context_budget is None:
            return {}
        return await context_budget.govern(state)

    async def plan(state: AgentState) -> dict[str, object]:
        approved = state.get("approved_action")
        if approved is not None:
            decision = PlanDecision.model_validate(approved)
        else:
            try:
                decision = await planner.plan(state)
            except AgentPlanningError:
                return {
                    "status": "failed",
                    "response": "对话规划服务暂不可用，请稍后重试。",
                    "error": "planner_unavailable",
                }
        return {"plan": decision.model_dump(mode="json")}

    async def execute_tools(state: AgentState) -> dict[str, object]:
        # A confirmation resume bypasses planning altogether.  The stored
        # pending action is still validated here and ToolRegistry remains the
        # authority for permission, argument, idempotency, and confirmation
        # enforcement before a handler can run.
        approved = state.get("approved_action")
        plan_data = PlanDecision.model_validate(
            approved if approved is not None else state.get("plan", {})
        )
        confirmed = approved is not None
        turn_id = state.get("turn_id", "")
        context = ToolContext(
            user_id=state["user_id"],
            workspace_id=state["workspace_id"],
            thread_id=state["thread_id"],
            graph_id=state.get("graph_id"),
            state=state,
            granted_permissions=frozenset(state.get("granted_permissions", [])),
            turn_id=turn_id,
            request_id=state.get("request_id", ""),
        )
        pending_calls = [
            _queued_tool_call(call, turn_id=turn_id) for call in plan_data.tool_calls
        ]
        results: list[dict[str, object]] = []
        for index, call in enumerate(plan_data.tool_calls):
            # Cancellation is checked immediately before every handler. A
            # running handler cannot be forcefully rolled back here; queued
            # calls are marked cancelled and are never submitted to Registry.
            if state.get("cancel_requested"):
                for queued in pending_calls[index:]:
                    queued["status"] = "cancelled"
                break
            pending_calls[index]["status"] = "running"
            execution_result = await tools.execute(
                call,
                context=context,
                confirmed=confirmed,
            )
            result_data = execution_result.model_dump(mode="json")
            result_data["turn_id"] = turn_id
            results.append(result_data)
            pending_calls[index]["operation_id"] = execution_result.operation_id
            pending_calls[index]["status"] = (
                "awaiting_confirmation"
                if execution_result.needs_confirmation
                else "cancelled"
                if execution_result.error_code == "cancelled"
                else "succeeded"
                if execution_result.ok
                else "failed"
            )
        history = list(state.get("tool_history", []))
        history.extend(results)
        history = _retain_recent_tool_rounds(history)
        tool_messages = list(state.get("messages", []))
        for result_data in results:
            tool_messages.append(
                {
                    "role": "tool",
                    "name": result_data["name"],
                    "content": result_data.get("output")
                    if result_data["ok"]
                    else result_data.get("error_message", "tool failed"),
                }
            )
        if state.get("cancel_requested"):
            return {
                "last_tool_results": results,
                "tool_history": history,
                "pending_tool_calls": pending_calls,
                "messages": tool_messages,
                "pending_action": None,
                "approved_action": None,
                "response": "已主动终止当前待执行操作。",
                "status": "cancelled",
            }
        if any(result["needs_confirmation"] for result in results):
            operation_ids = {
                result["call_id"]: result.get("operation_id") for result in results
            }
            pending_plan = plan_data.model_copy(
                update={
                    "tool_calls": [
                        call.model_copy(
                            update={"operation_id": operation_ids.get(call.call_id)}
                        )
                        for call in plan_data.tool_calls
                    ]
                }
            )
            return {
                "last_tool_results": results,
                "tool_history": history,
                "pending_tool_calls": pending_calls,
                "messages": tool_messages,
                "pending_action": pending_plan.model_dump(mode="json"),
                "approved_action": None,
                "response": "该操作需要用户确认后才能执行。",
                "status": "awaiting_confirmation",
            }
        return {
            "last_tool_results": results,
            "tool_history": history,
            "pending_tool_calls": pending_calls,
            "messages": tool_messages,
            "approved_action": None,
            "status": "tool_executed",
        }

    async def load_tool_details(state: AgentState) -> dict[str, object]:
        """Reveal whole schemas only after the planner has selected a tool."""

        attempts = int(state.get("tool_disclosure_attempts", 0))
        if attempts >= 3:
            return {
                "response": "工具定义无法在当前上下文预算内完整展开。",
                "status": "failed",
                "error": "tool_schema_context_exceeded",
            }
        plan_data = PlanDecision.model_validate(state.get("plan", {}))
        selected_names = plan_data.selected_tool_names
        details = tools.describe_selected(selected_names)
        exposed_names = {str(item.get("name")) for item in details}
        missing_names = [name for name in selected_names if name not in exposed_names]
        if missing_names:
            return {
                "response": "请求了不存在的工具定义。",
                "status": "failed",
                "error": f"unknown_tool_selection: {', '.join(missing_names)}",
            }
        return {
            "tool_details": details,
            "tool_disclosure_attempts": attempts + 1,
        }

    async def reject_undisclosed_tools(state: AgentState) -> dict[str, object]:
        """Reject calls whose full schema was never explicitly disclosed."""

        decision = PlanDecision.model_validate(state.get("plan", {}))
        detailed_names = {
            str(item.get("name")) for item in state.get("tool_details", [])
        }
        missing_names = sorted(
            {call.name for call in decision.tool_calls if call.name not in detailed_names}
        )
        return {
            "response": "工具调用必须先选择工具并读取完整定义。",
            "status": "failed",
            "error": f"tool_schema_not_disclosed: {', '.join(missing_names)}",
        }

    async def await_confirmation(state: AgentState) -> dict[str, object]:
        plan_data = PlanDecision.model_validate(state.get("plan", {}))
        pending = plan_data.model_dump(mode="json")
        return {
            "pending_action": pending,
            "response": plan_data.confirmation_message or "请确认是否执行上述操作。",
            "status": "awaiting_confirmation",
        }

    async def finalize(state: AgentState) -> dict[str, object]:
        existing = state.get("response")
        plan_data = PlanDecision.model_validate(state.get("plan", {}))
        response = existing or plan_data.message
        if not response:
            results = state.get("last_tool_results", [])
            response = "工具调用已完成。" if results else "请求已处理。"
        messages = list(state.get("messages", []))
        if not messages or messages[-1].get("content") != response:
            messages.append({"role": "assistant", "content": response})
        return {
            "messages": messages,
            "response": response,
            "status": state.get("status", "completed")
            if state.get("status") in {"cancelled", "max_steps", "failed"}
            else "completed",
        }

    def route_after_prepare(state: AgentState) -> str:
        if state.get("status") in {"cancelled", "max_steps"}:
            return "finalize"
        # User confirmation restores a server-stored action directly into the
        # executor. It deliberately does not make another model planning call
        # before the protected operation is submitted.
        if state.get("approved_action") is not None:
            return "execute_tools"
        return "load_context"

    def route_after_context_management(state: AgentState) -> str:
        return "finalize" if state.get("status") == "failed" else "plan"

    def route_after_plan(state: AgentState) -> str:
        if state.get("status") == "failed":
            return "finalize"
        decision = PlanDecision.model_validate(state.get("plan", {}))
        if decision.action == "select_tools":
            return "load_tool_details"
        if decision.action in {"tool", "confirm"}:
            detailed_names = {
                str(item.get("name")) for item in state.get("tool_details", [])
            }
            if any(call.name not in detailed_names for call in decision.tool_calls):
                return "reject_undisclosed_tools"
            return "execute_tools" if decision.action == "tool" else "await_confirmation"
        return "finalize"

    def route_after_tools(state: AgentState) -> str:
        if state.get("status") == "awaiting_confirmation":
            return "end"
        if state.get("status") == "cancelled":
            return "finalize"
        return "prepare"

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("load_context", load_context)
    graph.add_node("manage_context", manage_context)
    graph.add_node("plan", plan)
    graph.add_node("load_tool_details", load_tool_details)
    graph.add_node("reject_undisclosed_tools", reject_undisclosed_tools)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("await_confirmation", await_confirmation)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {
            "load_context": "load_context",
            "execute_tools": "execute_tools",
            "finalize": "finalize",
        },
    )
    graph.add_edge("load_context", "manage_context")
    graph.add_conditional_edges(
        "manage_context",
        route_after_context_management,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "load_tool_details": "load_tool_details",
            "reject_undisclosed_tools": "reject_undisclosed_tools",
            "execute_tools": "execute_tools",
            "await_confirmation": "await_confirmation",
            "finalize": "finalize",
        },
    )
    graph.add_edge("load_tool_details", "manage_context")
    graph.add_edge("reject_undisclosed_tools", "finalize")
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"end": END, "finalize": "finalize", "prepare": "prepare"},
    )
    graph.add_edge("await_confirmation", END)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
