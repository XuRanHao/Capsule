"""Application-facing wrapper around the compiled LangGraph Agent."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from capsule.agent.contracts import AgentRequest, AgentResponse
from capsule.agent.graph import AgentPlanner, ReadyPlanner, build_agent_graph
from capsule.agent.memory import AgentMemoryStore, NullMemoryStore
from capsule.agent.tools import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        *,
        planner: AgentPlanner | None = None,
        tools: ToolRegistry | None = None,
        memory: AgentMemoryStore | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        tool_permissions: frozenset[str] | None = None,
    ) -> None:
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = build_agent_graph(
            planner=planner or ReadyPlanner(),
            tools=tools or ToolRegistry(),
            memory=memory or NullMemoryStore(),
            checkpointer=self._checkpointer,
            tool_permissions=tool_permissions or frozenset(),
        )

    @property
    def graph(self):
        return self._graph

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        config = {"configurable": {"thread_id": request.thread_id}}
        state = await self._graph.ainvoke(
            {
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "graph_id": request.graph_id,
                "input_message": request.message,
                "confirmation_response": request.confirmation,
                "max_steps": request.max_steps,
            },
            config=config,
        )
        return AgentResponse(
            thread_id=request.thread_id,
            status=state.get("status", "failed"),
            message=state.get("response"),
            pending_action=state.get("pending_action"),
            tool_history=list(state.get("tool_history", [])),
            step_count=int(state.get("step_count", 0)),
        )

    async def state(self, thread_id: str) -> dict[str, Any] | None:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": thread_id}},
        )
        return dict(snapshot.values) if snapshot.values else None


def create_agent_runtime(
    *,
    planner: AgentPlanner | None = None,
    tools: ToolRegistry | None = None,
    memory: AgentMemoryStore | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    tool_permissions: frozenset[str] | None = None,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner,
        tools=tools,
        memory=memory,
        checkpointer=checkpointer,
        tool_permissions=tool_permissions,
    )
