"""LangGraph state machine used by the conversational Agent runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from capsule.agent.contracts import PlanDecision
from capsule.agent.memory import AgentMemoryStore
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


def build_agent_graph(
    *,
    planner: AgentPlanner,
    tools: ToolRegistry,
    memory: AgentMemoryStore,
    checkpointer: BaseCheckpointSaver,
):
    """Compile the resumable Agent graph with injected application dependencies."""

    async def prepare(state: AgentState) -> dict[str, object]:
        messages = list(state.get("messages", []))
        input_message = state.get("input_message")
        if input_message:
            messages.append({"role": "user", "content": input_message})

        updates: dict[str, object] = {
            "messages": messages,
            "input_message": None,
            "confirmation_response": None,
            "response": None,
            "error": None,
            "step_count": state.get("step_count", 0) + 1,
            "status": "planning",
        }
        pending = state.get("pending_action")
        confirmation = state.get("confirmation_response")
        if pending is not None and confirmation is not None:
            if confirmation:
                approved = dict(pending)
                approved["action"] = "tool"
                updates["approved_action"] = approved
                updates["pending_action"] = None
            else:
                updates["pending_action"] = None
                updates["approved_action"] = None
                updates["response"] = "已取消待执行操作。"
                updates["status"] = "cancelled"
        if int(updates["step_count"]) > int(state.get("max_steps", 8)):
            updates["response"] = "已达到本次会话的最大执行步数。"
            updates["status"] = "max_steps"
        return updates

    async def load_context(state: AgentState) -> dict[str, object]:
        messages = state.get("messages", [])
        query = str(messages[-1].get("content", "")) if messages else ""
        context = await memory.load(
            user_id=state["user_id"],
            workspace_id=state["workspace_id"],
            query=query,
        )
        return {"memory_context": context}

    async def plan(state: AgentState) -> dict[str, object]:
        approved = state.get("approved_action")
        if approved is not None:
            decision = PlanDecision.model_validate(approved)
        else:
            decision = await planner.plan(state)
        return {"plan": decision.model_dump(mode="json")}

    async def execute_tools(state: AgentState) -> dict[str, object]:
        plan_data = PlanDecision.model_validate(state.get("plan", {}))
        confirmed = state.get("approved_action") is not None
        context = ToolContext(
            user_id=state["user_id"],
            workspace_id=state["workspace_id"],
            thread_id=state["thread_id"],
            graph_id=state.get("graph_id"),
            state=state,
            granted_permissions=frozenset(state.get("granted_permissions", [])),
            turn_id=state.get("turn_id", ""),
        )
        results = []
        for call in plan_data.tool_calls:
            result = await tools.execute(call, context=context, confirmed=confirmed)
            result_data = result.model_dump(mode="json")
            result_data["turn_id"] = context.turn_id
            results.append(result_data)
        history = list(state.get("tool_history", []))
        history.extend(results)
        history = _retain_recent_tool_rounds(history)
        tool_messages = list(state.get("messages", []))
        for result in results:
            tool_messages.append(
                {
                    "role": "tool",
                    "name": result["name"],
                    "content": result.get("output")
                    if result["ok"]
                    else result.get("error_message", "tool failed"),
                }
            )
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
                "messages": tool_messages,
                "pending_action": pending_plan.model_dump(mode="json"),
                "approved_action": None,
                "response": "该操作需要用户确认后才能执行。",
                "status": "awaiting_confirmation",
            }
        return {
            "last_tool_results": results,
            "tool_history": history,
            "messages": tool_messages,
            "approved_action": None,
            "status": "tool_executed",
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
            if state.get("status") in {"cancelled", "max_steps"}
            else "completed",
        }

    async def persist_memory(state: AgentState) -> dict[str, object]:
        plan_data = PlanDecision.model_validate(state.get("plan", {}))
        if plan_data.memory_writes:
            await memory.save(
                user_id=state["user_id"],
                workspace_id=state["workspace_id"],
                writes=plan_data.memory_writes,
            )
        return {}

    def route_after_prepare(state: AgentState) -> str:
        return "finalize" if state.get("status") in {"cancelled", "max_steps"} else "load_context"

    def route_after_plan(state: AgentState) -> str:
        decision = PlanDecision.model_validate(state.get("plan", {}))
        if decision.action == "tool" and decision.tool_calls:
            return "execute_tools"
        if decision.action == "confirm":
            return "await_confirmation"
        return "finalize"

    def route_after_tools(state: AgentState) -> str:
        return "end" if state.get("status") == "awaiting_confirmation" else "prepare"

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("load_context", load_context)
    graph.add_node("plan", plan)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("await_confirmation", await_confirmation)
    graph.add_node("finalize", finalize)
    graph.add_node("persist_memory", persist_memory)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"load_context": "load_context", "finalize": "finalize"},
    )
    graph.add_edge("load_context", "plan")
    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "execute_tools": "execute_tools",
            "await_confirmation": "await_confirmation",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {"end": END, "prepare": "prepare"},
    )
    graph.add_edge("await_confirmation", END)
    graph.add_edge("finalize", "persist_memory")
    graph.add_edge("persist_memory", END)
    return graph.compile(checkpointer=checkpointer)
