"""Application-facing wrapper around the compiled LangGraph Agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from capsule.agent.context_budget import (
    ContextBudget,
    ContextBudgetController,
    MemoryEventPublisher,
)
from capsule.agent.contracts import AgentRequest, AgentResponse
from capsule.agent.graph import AgentPlanner, ReadyPlanner, build_agent_graph
from capsule.agent.memory import AgentMemoryStore, DelegatingMemoryStore, NullMemoryStore
from capsule.agent.tools import ToolRegistry
from capsule.db.agent_memory import (
    AgentConversationRepository,
    ConversationContext,
    ConversationSyncState,
)

PermissionLoader = Callable[..., Awaitable[Iterable[str]]]


class AgentRuntime:
    def __init__(
        self,
        *,
        planner: AgentPlanner | None = None,
        tools: ToolRegistry | None = None,
        memory: AgentMemoryStore | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        permission_loader: PermissionLoader | None = None,
        conversation_repository: AgentConversationRepository | None = None,
        context_budget: ContextBudget | None = None,
        memory_event_publisher: MemoryEventPublisher | None = None,
        turn_lease_seconds: float = 300.0,
    ) -> None:
        if turn_lease_seconds <= 0:
            raise ValueError("turn lease duration must be positive")
        self._checkpointer = checkpointer or InMemorySaver()
        self._planner = planner or ReadyPlanner()
        self._tools = tools or ToolRegistry()
        self._memory_store = DelegatingMemoryStore(memory or NullMemoryStore())
        self._permission_loader = permission_loader
        self._conversation_repository = conversation_repository
        self._context_budget = context_budget or ContextBudget()
        self._memory_event_publisher = memory_event_publisher
        self._turn_lease_seconds = turn_lease_seconds
        self._context_budget_controller = ContextBudgetController(
            budget=self._context_budget,
            repository=conversation_repository,
            publish_event=memory_event_publisher,
        )
        self._graph = self._build_graph()

    def set_permission_loader(self, loader: PermissionLoader) -> None:
        """Attach a server-side permission loader after app startup."""

        self._permission_loader = loader

    def set_conversation_repository(
        self,
        repository: AgentConversationRepository,
        *,
        context_budget: ContextBudget | None = None,
        memory_event_publisher: MemoryEventPublisher | None = None,
        turn_lease_seconds: float | None = None,
    ) -> None:
        """Attach durable conversation storage after application database startup."""

        self._conversation_repository = repository
        if context_budget is not None:
            self._context_budget = context_budget
        if turn_lease_seconds is not None:
            if turn_lease_seconds <= 0:
                raise ValueError("turn lease duration must be positive")
            self._turn_lease_seconds = turn_lease_seconds
        self._memory_event_publisher = memory_event_publisher
        self._context_budget_controller = ContextBudgetController(
            budget=self._context_budget,
            repository=repository,
            publish_event=memory_event_publisher,
        )
        self._graph = self._build_graph()

    def set_memory_store(self, memory: AgentMemoryStore) -> None:
        """Swap the durable reader without rebuilding active graph checkpoints."""

        self._memory_store.set_delegate(memory)

    def set_checkpointer(self, checkpointer: BaseCheckpointSaver[Any]) -> None:
        """Install a durable saver during application startup before any invocation."""

        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    @property
    def graph(self) -> Any:
        return self._graph

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        conversation = self._conversation_repository
        turn_lease_owner: str | None = None
        turn_lease_lost = asyncio.Event()
        turn_lease_heartbeat: asyncio.Task[None] | None = None
        if conversation is not None:
            turn_lease_owner = f"agent-turn-{uuid4().hex}"
            await conversation.acquire_turn_lease(
                thread_id=request.thread_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                owner=turn_lease_owner,
                lease_seconds=self._turn_lease_seconds,
            )
            turn_lease_heartbeat = asyncio.create_task(
                self._renew_turn_lease_until_done(
                    repository=conversation,
                    request=request,
                    owner=turn_lease_owner,
                    lease_lost=turn_lease_lost,
                )
            )
        try:
            return await self._invoke_under_turn_lease(
                request,
                turn_lease_lost=turn_lease_lost,
            )
        finally:
            if turn_lease_heartbeat is not None:
                turn_lease_heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await turn_lease_heartbeat
            if conversation is not None and turn_lease_owner is not None:
                await conversation.release_turn_lease(
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    owner=turn_lease_owner,
                )

    async def _invoke_under_turn_lease(
        self,
        request: AgentRequest,
        *,
        turn_lease_lost: asyncio.Event,
    ) -> AgentResponse:
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
        durable_context: dict[str, object] = {}
        conversation = self._conversation_repository
        if conversation is not None:
            sync_state: ConversationSyncState | None = None
            if previous:
                if not resumes_pending:
                    sync_state = await conversation.get_sync_state(
                        thread_id=request.thread_id,
                        user_id=request.user_id,
                        workspace_id=request.workspace_id,
                    )
            persisted_user = await conversation.append_message(
                thread_id=request.thread_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                role="user",
                content=request.message,
                request_id=request_id,
                turn_id=turn_id,
            )
            if not previous:
                durable = await conversation.get_context(
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    max_messages=None,
                )
                durable_context = {
                    "messages": [
                        item.to_graph_message()
                        for item in durable.messages
                        if item.message_id != persisted_user.message_id
                    ],
                    "working_context": _working_context(
                        durable,
                        hot_mirror_through_sequence=persisted_user.sequence - 1,
                    ),
                }
            elif (
                not resumes_pending
                and sync_state is not None
                and _requires_hot_mirror_refresh(
                    previous,
                    sync_state=sync_state,
                    persisted_user_sequence=persisted_user.sequence,
                )
            ):
                # A different process may have advanced this thread, or the
                # memory Worker may have replaced its summary.  Rehydrate only
                # when one of the two durable watermarks no longer matches.
                durable = await conversation.get_context(
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    max_messages=None,
                )
                await self._graph.aupdate_state(
                    config,
                    {
                        "messages": [
                            item.to_graph_message()
                            for item in durable.messages
                            if item.message_id != persisted_user.message_id
                        ],
                        "working_context": _working_context(
                            durable,
                            hot_mirror_through_sequence=persisted_user.sequence - 1,
                        ),
                    },
                )
        granted_permissions: frozenset[str] = frozenset()
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
                # Resuming confirmation executes the stored action before the
                # next planner pass. Preserve the earlier frozen recall so a
                # background memory Worker cannot change that turn's result.
                "memory_context_request_id": (
                    request_id
                    if resumes_pending and "memory_context" in previous
                    else None
                ),
                **durable_context,
            },
            config=config,
        )
        if turn_lease_lost.is_set():
            raise RuntimeError("agent turn lease was lost before the response completed")
        if conversation is not None and state.get("response") is not None:
            persisted_assistant = await conversation.append_message(
                thread_id=request.thread_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                role="assistant",
                content=state["response"],
                request_id=request_id,
                turn_id=turn_id,
            )
            await self._graph.aupdate_state(
                config,
                {
                    "working_context": _with_hot_mirror_sequence(
                        state.get("working_context"),
                        persisted_assistant.sequence,
                    )
                },
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

    async def _renew_turn_lease_until_done(
        self,
        *,
        repository: AgentConversationRepository,
        request: AgentRequest,
        owner: str,
        lease_lost: asyncio.Event,
    ) -> None:
        """Keep a long model/tool turn exclusive without holding a DB transaction."""

        interval = min(30.0, max(0.1, self._turn_lease_seconds / 3))
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await repository.renew_turn_lease(
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    owner=owner,
                    lease_seconds=self._turn_lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # A transient database error should not end a model call;
                # another heartbeat may renew before the durable lease ends.
                continue
            if not renewed:
                lease_lost.set()
                return

    async def state(self, thread_id: str) -> dict[str, Any] | None:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": thread_id}},
        )
        return dict(snapshot.values) if snapshot.values else None

    async def discard_state(self, thread_id: str) -> None:
        """Remove a suspended graph so archived or deleted threads cannot resume it."""

        await self._checkpointer.adelete_thread(thread_id)

    def _build_graph(self) -> Any:
        return build_agent_graph(
            planner=self._planner,
            tools=self._tools,
            memory=self._memory_store,
            checkpointer=self._checkpointer,
            context_budget=self._context_budget_controller,
        )


def create_agent_runtime(
    *,
    planner: AgentPlanner | None = None,
    tools: ToolRegistry | None = None,
    memory: AgentMemoryStore | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    permission_loader: PermissionLoader | None = None,
    conversation_repository: AgentConversationRepository | None = None,
    context_budget: ContextBudget | None = None,
    memory_event_publisher: MemoryEventPublisher | None = None,
    turn_lease_seconds: float = 300.0,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner,
        tools=tools,
        memory=memory,
        checkpointer=checkpointer,
        permission_loader=permission_loader,
        conversation_repository=conversation_repository,
        context_budget=context_budget,
        memory_event_publisher=memory_event_publisher,
        turn_lease_seconds=turn_lease_seconds,
    )


def _working_context(
    context: ConversationContext,
    *,
    hot_mirror_through_sequence: int,
) -> dict[str, Any]:
    return {
        "conversation_summary": context.thread.summary,
        "conversation_topic": context.thread.summary_topic,
        "summary_covered_sequence": context.thread.summary_covered_sequence,
        "memory_revision": context.thread.memory_revision,
        "hot_mirror_through_sequence": hot_mirror_through_sequence,
    }


def _memory_revision(state: dict[str, Any]) -> int:
    working_context = state.get("working_context")
    if not isinstance(working_context, dict):
        return 0
    revision = working_context.get("memory_revision")
    return revision if isinstance(revision, int) and revision >= 0 else 0


def _requires_hot_mirror_refresh(
    state: dict[str, Any],
    *,
    sync_state: ConversationSyncState,
    persisted_user_sequence: int,
) -> bool:
    """Decide whether cached graph messages still describe the durable thread."""

    mirrored_sequence = _hot_mirror_sequence(state)
    return (
        mirrored_sequence != sync_state.last_message_sequence
        or mirrored_sequence != persisted_user_sequence - 1
        or sync_state.memory_revision > _memory_revision(state)
    )


def _hot_mirror_sequence(state: dict[str, Any]) -> int | None:
    working_context = state.get("working_context")
    if not isinstance(working_context, dict):
        return None
    value = working_context.get("hot_mirror_through_sequence")
    return value if isinstance(value, int) and value >= 0 else None


def _with_hot_mirror_sequence(
    working_context: object,
    sequence: int,
) -> dict[str, Any]:
    context = dict(working_context) if isinstance(working_context, dict) else {}
    context["hot_mirror_through_sequence"] = sequence
    return context
