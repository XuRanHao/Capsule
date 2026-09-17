"""Application-facing wrapper around the compiled LangGraph Agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

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
        conversation_context_messages: int = 24,
        memory_consolidation_token_threshold: int = 4_000,
    ) -> None:
        if conversation_context_messages < 1:
            raise ValueError("conversation_context_messages must be positive")
        if memory_consolidation_token_threshold < 1:
            raise ValueError("memory_consolidation_token_threshold must be positive")
        self._checkpointer = checkpointer or InMemorySaver()
        self._memory_store = DelegatingMemoryStore(memory or NullMemoryStore())
        self._permission_loader = permission_loader
        self._conversation_repository = conversation_repository
        self._conversation_context_messages = conversation_context_messages
        self._memory_consolidation_token_threshold = memory_consolidation_token_threshold
        self._graph = build_agent_graph(
            planner=planner or ReadyPlanner(),
            tools=tools or ToolRegistry(),
            memory=self._memory_store,
            checkpointer=self._checkpointer,
        )

    def set_permission_loader(self, loader: PermissionLoader) -> None:
        """Attach a server-side permission loader after app startup."""

        self._permission_loader = loader

    def set_conversation_repository(
        self,
        repository: AgentConversationRepository,
        *,
        context_messages: int,
        consolidation_token_threshold: int,
    ) -> None:
        """Attach durable conversation storage after application database startup."""

        if context_messages < 1 or consolidation_token_threshold < 1:
            raise ValueError("conversation limits must be positive")
        self._conversation_repository = repository
        self._conversation_context_messages = context_messages
        self._memory_consolidation_token_threshold = consolidation_token_threshold

    def set_memory_store(self, memory: AgentMemoryStore) -> None:
        """Swap the durable reader without rebuilding active graph checkpoints."""

        self._memory_store.set_delegate(memory)

    @property
    def graph(self) -> Any:
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
        durable_context: dict[str, object] = {}
        conversation = self._conversation_repository
        if conversation is not None:
            sync_state: ConversationSyncState | None = None
            if previous:
                # A hot in-process conversation is still the fastest context source.
                await conversation.enqueue_consolidation_if_needed(
                    thread_id=request.thread_id,
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    token_threshold=self._memory_consolidation_token_threshold,
                )
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
                    max_messages=self._conversation_context_messages + 1,
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
                    max_messages=self._conversation_context_messages + 1,
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
                **durable_context,
            },
            config=config,
        )
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
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    permission_loader: PermissionLoader | None = None,
    conversation_repository: AgentConversationRepository | None = None,
    conversation_context_messages: int = 24,
    memory_consolidation_token_threshold: int = 4_000,
) -> AgentRuntime:
    return AgentRuntime(
        planner=planner,
        tools=tools,
        memory=memory,
        checkpointer=checkpointer,
        permission_loader=permission_loader,
        conversation_repository=conversation_repository,
        conversation_context_messages=conversation_context_messages,
        memory_consolidation_token_threshold=memory_consolidation_token_threshold,
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
