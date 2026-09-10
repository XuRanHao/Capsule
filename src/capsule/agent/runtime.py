"""Application-facing wrapper around the compiled LangGraph Agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from capsule.agent.contracts import AgentRequest, AgentResponse
from capsule.agent.graph import AgentPlanner, ReadyPlanner, build_agent_graph
from capsule.agent.memory import AgentMemoryStore, NullMemoryStore
from capsule.agent.tools import ToolRegistry

PermissionLoader = Callable[..., Awaitable[Iterable[str]]]


class AgentRuntime:
    def __init__(
        self,
        *,
        planner: AgentPlanner | None = None,
        tools: ToolRegistry | None = None,
        memory: AgentMemoryStore | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        permission_loader: PermissionLoader | None = None,
    ) -> None:
        self._checkpointer = checkpointer or InMemorySaver()
        self._permission_loader = permission_loader
        self._graph = build_agent_graph(
            planner=planner or ReadyPlanner(),
            tools=tools or ToolRegistry(),
            memory=memory or NullMemoryStore(),
            checkpointer=self._checkpointer,
        )

    def set_permission_loader(self, loader: PermissionLoader) -> None:
        """Attach a server-side permission loader after app startup."""

        self._permission_loader = loader

    @property
    def graph(self):
        return self._graph

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        config = {"configurable": {"thread_id": request.thread_id}}
        snapshot = await self._graph.aget_state(config)
        previous = dict(snapshot.values) if snapshot.values else {}
        # Confirmation/cancellation is a continuation of the turn that
        # produced the pending action.  A normal message starts a new turn
        # and therefore clears stale queued calls in the graph state.
        resumes_pending = bool(previous.get("pending_action")) and (
            request.confirmation is not None or request.cancel
        )
        turn_id = (
            str(previous.get("turn_id"))
            if resumes_pending and previous.get("turn_id")
            else f"turn_{uuid4().hex}"
        )
        request_id = request.request_id or f"request_{uuid4().hex}"
        granted_permissions = frozenset()
        if self._permission_loader is not None:
            granted_permissions = frozenset(
                await self._permission_loader(
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                )
            )
        state = await self._graph.ainvoke(
            {
                "thread_id": request.thread_id,
                # A complete runtime invocation is one Agent output round.
                # Tool loops inside this invocation keep the same turn ID.
                "turn_id": turn_id,
                "request_id": request_id,
                "start_new_turn": not resumes_pending,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "graph_id": request.graph_id,
                "granted_permissions": sorted(granted_permissions),
                "input_message": request.message,
                "confirmation_response": request.confirmation,
                "cancel_requested": request.cancel,
                "max_steps": request.max_steps,
            },
            config=config,
        )
        return AgentResponse(
            thread_id=request.thread_id,
            status=state.get("status", "failed"),
            message=state.get("response"),
            pending_action=state.get("pending_action"),
            pending_tool_calls=list(state.get("pending_tool_calls", [])),
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
    permission_loader: PermissionLoader | None = None,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner,
        tools=tools,
        memory=memory,
        checkpointer=checkpointer,
        permission_loader=permission_loader,
    )
