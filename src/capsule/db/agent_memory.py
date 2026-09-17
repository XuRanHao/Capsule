"""PostgreSQL repositories for Agent conversations and memory consolidation."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.agent.contracts import MemoryWrite
from capsule.agent.memory_contracts import (
    ConversationSummary,
    MemoryCandidate,
    MemoryConsolidation,
    MemoryMutation,
)
from capsule.db.models import (
    AgentMemory,
    AgentMemoryOutbox,
    AgentMemorySource,
    AgentMemoryVectorOutbox,
    AgentMessage,
    AgentThread,
    Workspace,
    WorkspaceMemoryProfile,
)
from capsule.db.session import Database

MessageRole = Literal["user", "assistant", "tool", "system"]
ThreadStatus = Literal["active", "archived", "deleted"]


class AgentThreadStateError(ValueError):
    """The identified conversation exists but cannot accept this operation."""


@dataclass(frozen=True, slots=True)
class AgentMessageRecord:
    message_id: str
    thread_id: str
    sequence: int
    role: str
    content: Any
    name: str | None
    turn_id: str | None
    request_id: str | None
    estimated_tokens: int
    created_at: datetime

    def to_graph_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            message["name"] = self.name
        return message


@dataclass(frozen=True, slots=True)
class AgentThreadRecord:
    thread_id: str
    user_id: str
    workspace_id: str
    title: str
    status: str
    summary: str | None
    summary_topic: str | None
    summary_covered_sequence: int
    last_consolidated_sequence: int
    memory_revision: int
    last_message_at: datetime | None
    last_message_sequence: int = 0
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ConversationContext:
    thread: AgentThreadRecord
    messages: list[AgentMessageRecord]


@dataclass(frozen=True, slots=True)
class ConversationSyncState:
    """Small authoritative marker used to validate an in-process session mirror."""

    last_message_sequence: int
    memory_revision: int


@dataclass(frozen=True, slots=True)
class MemoryOutboxEvent:
    event_id: str
    thread_id: str
    user_id: str
    workspace_id: str
    through_sequence: int


@dataclass(frozen=True, slots=True)
class MemoryVectorOutboxEvent:
    event_id: str
    memory_id: str
    memory_version: int


@dataclass(frozen=True, slots=True)
class AgentMemoryVectorDocument:
    """Current PostgreSQL truth for one derived Milvus vector."""

    memory_id: str
    active: bool
    scope: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    kind: str | None = None
    memory_key: str | None = None
    display_text: str | None = None
    topics: list[str] | None = None
    version: int | None = None


@dataclass(frozen=True, slots=True)
class ClaimedMemoryConsolidation:
    event: MemoryOutboxEvent
    messages: list[AgentMessageRecord]
    thread: AgentThreadRecord
    active_topics: list[dict[str, Any]]


MemoryClaimStatus = Literal["claimed", "already_consolidated", "lease_held"]
MemoryVectorClaimStatus = Literal["claimed", "already_completed", "lease_held"]


@dataclass(frozen=True, slots=True)
class MemoryConsolidationClaim:
    """The durable disposition of one Worker attempt to acquire a thread lease."""

    status: MemoryClaimStatus
    consolidation: ClaimedMemoryConsolidation | None = None


@dataclass(frozen=True, slots=True)
class MemoryVectorIndexClaim:
    """The durable disposition of one vector-indexing delivery."""

    status: MemoryVectorClaimStatus
    document: AgentMemoryVectorDocument | None = None


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    memory_id: str
    scope: str
    kind: str
    memory_key: str
    value: dict[str, Any]
    display_text: str
    topics: list[str]
    confidence: float
    effective_confidence: float
    decay_rate: float
    status: str
    relevance: float


class AgentConversationRepository:
    """Own conversation history, threshold events, and memory-worker fences."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def create_thread(
        self,
        *,
        user_id: str,
        workspace_id: str,
        title: str | None = None,
    ) -> AgentThreadRecord:
        async with self._database.session() as session, session.begin():
            workspace = await session.get(Workspace, workspace_id)
            if workspace is None:
                raise ValueError(f"workspace does not exist: {workspace_id}")
            thread = AgentThread(user_id=user_id, workspace_id=workspace_id)
            if title is not None:
                thread.title = title
            session.add(thread)
            await session.flush()
            return _thread_record(thread)

    async def rename_thread(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        title: str,
    ) -> AgentThreadRecord:
        """Change the visible title without changing the durable message history."""

        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("thread title must not be blank")
        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            self._require_not_deleted(thread)
            thread.title = normalized_title
            await session.flush()
            return _thread_record(thread)

    async def archive_thread(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
    ) -> AgentThreadRecord:
        """Hide an active conversation while preserving its durable history."""

        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            if thread.status == "archived":
                return _thread_record(thread)
            if thread.status != "active":
                raise AgentThreadStateError(
                    "only an active agent thread can be archived"
                )
            thread.status = "archived"
            await session.flush()
            return _thread_record(thread)

    async def restore_thread(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
    ) -> AgentThreadRecord:
        """Restore an archived or softly deleted conversation as a new active session."""

        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            thread.status = "active"
            thread.deleted_at = None
            thread.memory_lease_owner = None
            thread.memory_lease_expires_at = None
            await session.flush()
            return _thread_record(thread)

    async def soft_delete_thread(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
    ) -> AgentThreadRecord:
        """Hide a conversation and prevent its queued history from creating new memory."""

        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            if thread.status == "deleted":
                return _thread_record(thread)
            thread.status = "deleted"
            thread.deleted_at = datetime.now(UTC)
            thread.memory_lease_owner = None
            thread.memory_lease_expires_at = None
            await session.execute(
                delete(AgentMemoryOutbox).where(
                    AgentMemoryOutbox.thread_id == thread_id
                )
            )
            await session.flush()
            return _thread_record(thread)

    async def append_message(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        role: MessageRole,
        content: Any,
        request_id: str | None = None,
        turn_id: str | None = None,
        name: str | None = None,
        estimated_tokens: int | None = None,
    ) -> AgentMessageRecord:
        """Append one message or return the idempotent existing message."""

        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=True,
            )
            self._require_active(thread)
            if request_id is not None:
                existing = await session.scalar(
                    select(AgentMessage).where(
                        AgentMessage.thread_id == thread_id,
                        AgentMessage.role == role,
                        AgentMessage.request_id == request_id,
                    )
                )
                if existing is not None:
                    return _message_record(existing)
            next_sequence = thread.last_message_sequence + 1
            now = datetime.now(UTC)
            message = AgentMessage(
                thread_id=thread_id,
                sequence=next_sequence,
                turn_id=turn_id,
                request_id=request_id,
                role=role,
                name=name,
                content=content,
                estimated_tokens=(
                    estimated_tokens
                    if estimated_tokens is not None
                    else estimate_message_tokens(content)
                ),
            )
            session.add(message)
            thread.last_message_sequence = next_sequence
            thread.last_message_at = now
            await session.flush()
            return _message_record(message)

    async def get_context(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        max_messages: int | None,
    ) -> ConversationContext:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be positive")
        async with self._database.session() as session:
            thread = await self._thread_for_identity(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            statement = (
                select(AgentMessage)
                .where(
                    AgentMessage.thread_id == thread_id,
                    AgentMessage.sequence > thread.summary_covered_sequence,
                )
                .order_by(AgentMessage.sequence.desc())
            )
            if max_messages is not None:
                statement = statement.limit(max_messages)
            rows = list(await session.scalars(statement))
        rows.reverse()
        return ConversationContext(
            thread=_thread_record(thread),
            messages=[_message_record(row) for row in rows],
        )

    async def get_sync_state(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
    ) -> ConversationSyncState:
        """Read only the durable markers needed before reusing hot graph state."""

        async with self._database.session() as session:
            thread = await self._thread_for_identity(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            self._require_active(thread)
            return ConversationSyncState(
                last_message_sequence=thread.last_message_sequence,
                memory_revision=thread.memory_revision,
            )

    async def list_messages(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[AgentMessageRecord]:
        if after_sequence < 0:
            raise ValueError("after_sequence must not be negative")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        async with self._database.session() as session:
            thread = await self._thread_for_identity(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            self._require_not_deleted(thread)
            rows = list(
                await session.scalars(
                    select(AgentMessage)
                    .where(
                        AgentMessage.thread_id == thread_id,
                        AgentMessage.sequence > after_sequence,
                    )
                    .order_by(AgentMessage.sequence)
                    .limit(limit)
                )
            )
        return [_message_record(row) for row in rows]

    async def list_threads(
        self,
        *,
        user_id: str,
        workspace_id: str,
        limit: int = 50,
        status: ThreadStatus | None = None,
        query: str | None = None,
        include_deleted: bool = False,
    ) -> list[AgentThreadRecord]:
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        search_title = query.strip() if query is not None else ""
        filters = [
            AgentThread.user_id == user_id,
            AgentThread.workspace_id == workspace_id,
        ]
        if status is not None:
            filters.append(AgentThread.status == status)
        elif not include_deleted:
            filters.append(AgentThread.status != "deleted")
        if search_title:
            escaped = (
                search_title.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            filters.append(AgentThread.title.ilike(f"%{escaped}%", escape="\\"))
        async with self._database.session() as session:
            rows = list(
                await session.scalars(
                    select(AgentThread)
                    .where(*filters)
                    .order_by(AgentThread.updated_at.desc())
                    .limit(limit)
                )
            )
        return [_thread_record(row) for row in rows]

    async def enqueue_consolidation_if_needed(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        token_threshold: int,
    ) -> MemoryOutboxEvent | None:
        """Durably request consolidation when the unprocessed range is large enough."""

        if token_threshold < 1:
            raise ValueError("token_threshold must be positive")
        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            self._require_active(thread)
            pending_tokens = await session.scalar(
                select(func.coalesce(func.sum(AgentMessage.estimated_tokens), 0)).where(
                    AgentMessage.thread_id == thread_id,
                    AgentMessage.sequence > thread.last_consolidated_sequence,
                )
            )
            if int(pending_tokens or 0) < token_threshold:
                return None
            through_sequence = await session.scalar(
                select(func.coalesce(func.max(AgentMessage.sequence), 0)).where(
                    AgentMessage.thread_id == thread_id
                )
            )
            resolved_through_sequence = int(through_sequence or 0)
            if resolved_through_sequence <= thread.last_consolidated_sequence:
                return None
            existing = await session.scalar(
                select(AgentMemoryOutbox).where(
                    AgentMemoryOutbox.thread_id == thread_id,
                    AgentMemoryOutbox.through_sequence == resolved_through_sequence,
                )
            )
            if existing is not None:
                return _outbox_event(existing)
            event = AgentMemoryOutbox(
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                through_sequence=resolved_through_sequence,
            )
            session.add(event)
            await session.flush()
            return _outbox_event(event)

    async def enqueue_consolidation(
        self,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        through_sequence: int,
    ) -> MemoryOutboxEvent | None:
        """Durably request a specific old-message prefix for compaction.

        The caller chooses the cutoff after a complete context preflight.  No
        ordinary message-count or token threshold is consulted here: this path
        exists solely for an over-budget context that needs a new summary.
        """

        if through_sequence < 1:
            raise ValueError("through_sequence must be positive")
        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                create=False,
            )
            self._require_active(thread)
            if through_sequence <= thread.last_consolidated_sequence:
                return None
            if through_sequence > thread.last_message_sequence:
                raise ValueError("through_sequence is ahead of conversation messages")
            existing = await session.scalar(
                select(AgentMemoryOutbox).where(
                    AgentMemoryOutbox.thread_id == thread_id,
                    AgentMemoryOutbox.through_sequence == through_sequence,
                )
            )
            if existing is not None:
                return _outbox_event(existing)
            event = AgentMemoryOutbox(
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                through_sequence=through_sequence,
            )
            session.add(event)
            await session.flush()
            return _outbox_event(event)

    async def pending_outbox_events(self, *, limit: int = 100) -> list[MemoryOutboxEvent]:
        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        now = datetime.now(UTC)
        async with self._database.session() as session:
            rows = list(
                await session.scalars(
                    select(AgentMemoryOutbox)
                    .where(
                        AgentMemoryOutbox.status == "pending",
                        AgentMemoryOutbox.available_at <= now,
                    )
                    .order_by(AgentMemoryOutbox.created_at)
                    .limit(limit)
                )
            )
        return [_outbox_event(row) for row in rows]

    async def pending_vector_outbox_events(
        self,
        *,
        limit: int = 100,
    ) -> list[MemoryVectorOutboxEvent]:
        """Return committed but not-yet-published Milvus synchronization events."""

        if limit < 1 or limit > 1_000:
            raise ValueError("limit must be between 1 and 1000")
        now = datetime.now(UTC)
        async with self._database.session() as session:
            rows = list(
                await session.scalars(
                    select(AgentMemoryVectorOutbox)
                    .where(
                        AgentMemoryVectorOutbox.status == "pending",
                        AgentMemoryVectorOutbox.available_at <= now,
                    )
                    .order_by(AgentMemoryVectorOutbox.created_at)
                    .limit(limit)
                )
            )
        return [_vector_outbox_event(row) for row in rows]

    async def retrieve_memory_matches(
        self,
        *,
        user_id: str,
        workspace_id: str,
        candidate: MemoryCandidate,
        limit: int = 3,
    ) -> list[MemoryMatch]:
        """Retrieve the top existing memories for one proposed-memory RAG pass."""

        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        filters = [
            AgentMemory.scope == candidate.scope,
            AgentMemory.user_id == user_id,
            AgentMemory.status == "active",
        ]
        if candidate.scope == "workspace":
            filters.append(AgentMemory.workspace_id == workspace_id)
        else:
            filters.append(AgentMemory.workspace_id.is_(None))
        exact_statement = select(AgentMemory).where(
            *filters,
            AgentMemory.kind == candidate.kind,
            AgentMemory.memory_key == candidate.memory_key,
        )
        text_statement = (
            select(
                AgentMemory,
                func.pdb.score(AgentMemory.memory_id).label("bm25_score"),
            )
            .where(*filters, AgentMemory.memory_id.op("@@@")(candidate.display_text))
            .order_by(
                func.pdb.score(AgentMemory.memory_id).desc(),
                AgentMemory.memory_id.asc(),
            )
            .limit(max(limit * 4, 8))
        )
        async with self._database.session() as session:
            exact = list(await session.scalars(exact_statement))
            text_rows = (await session.execute(text_statement)).all()
        bm25_by_id = {memory.memory_id: float(score) for memory, score in text_rows}
        by_id = {memory.memory_id: memory for memory in exact}
        by_id.update({memory.memory_id: memory for memory, _ in text_rows})
        maximum_bm25 = max(bm25_by_id.values(), default=0.0)
        now = datetime.now(UTC)
        matches = [
            _memory_match(
                memory,
                exact_key=(
                    memory.kind == candidate.kind
                    and memory.memory_key == candidate.memory_key
                ),
                bm25_score=bm25_by_id.get(memory.memory_id, 0.0),
                maximum_bm25=maximum_bm25,
                now=now,
            )
            for memory in by_id.values()
        ]
        matches.sort(
            key=lambda item: (item.relevance, item.memory_id),
            reverse=True,
        )
        return matches[:limit]

    async def retrieve_memory_context(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
        per_scope_limit: int,
    ) -> list[dict[str, Any]]:
        """Recall a bounded context set, keeping workspace memory before global memory."""

        if per_scope_limit < 1 or per_scope_limit > 10:
            raise ValueError("per_scope_limit must be between 1 and 10")
        normalized_query = query.strip()
        if not normalized_query:
            return []
        workspace_matches, global_matches = await asyncio.gather(
            self._retrieve_memory_scope(
                scope="workspace",
                user_id=user_id,
                workspace_id=workspace_id,
                query=normalized_query,
                limit=per_scope_limit,
            ),
            self._retrieve_memory_scope(
                scope="global",
                user_id=user_id,
                workspace_id=workspace_id,
                query=normalized_query,
                limit=per_scope_limit,
            ),
        )
        return [
            *[_agent_memory_context(item) for item in workspace_matches],
            *[_agent_memory_context(item) for item in global_matches],
        ]

    async def retrieve_memory_context_from_vectors(
        self,
        *,
        user_id: str,
        workspace_id: str,
        workspace_candidates: Sequence[tuple[str, float, int]],
        global_candidates: Sequence[tuple[str, float, int]],
        per_scope_limit: int,
    ) -> list[dict[str, Any]]:
        """Hydrate ANN candidate IDs through PostgreSQL before context injection.

        Milvus only provides a best-effort candidate set.  PostgreSQL applies the
        current user/workspace/status boundary and rejects an older vector version
        before confidence and time decay decide the final order.
        """

        if per_scope_limit < 1 or per_scope_limit > 10:
            raise ValueError("per_scope_limit must be between 1 and 10")
        workspace_matches, global_matches = await asyncio.gather(
            self._retrieve_memory_vector_scope(
                scope="workspace",
                user_id=user_id,
                workspace_id=workspace_id,
                candidates=workspace_candidates,
                limit=per_scope_limit,
            ),
            self._retrieve_memory_vector_scope(
                scope="global",
                user_id=user_id,
                workspace_id=workspace_id,
                candidates=global_candidates,
                limit=per_scope_limit,
            ),
        )
        return [
            *[_agent_memory_context(item) for item in workspace_matches],
            *[_agent_memory_context(item) for item in global_matches],
        ]

    async def _retrieve_memory_scope(
        self,
        *,
        scope: Literal["workspace", "global"],
        user_id: str,
        workspace_id: str,
        query: str,
        limit: int,
    ) -> list[MemoryMatch]:
        filters = [
            AgentMemory.scope == scope,
            AgentMemory.user_id == user_id,
            AgentMemory.status == "active",
        ]
        if scope == "workspace":
            filters.append(AgentMemory.workspace_id == workspace_id)
        else:
            filters.append(AgentMemory.workspace_id.is_(None))
        statement = (
            select(
                AgentMemory,
                func.pdb.score(AgentMemory.memory_id).label("bm25_score"),
            )
            .where(*filters, AgentMemory.memory_id.op("@@@")(query))
            .order_by(
                func.pdb.score(AgentMemory.memory_id).desc(),
                AgentMemory.memory_id.asc(),
            )
            .limit(max(limit * 4, 8))
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        bm25_by_id = {memory.memory_id: float(score) for memory, score in rows}
        maximum_bm25 = max(bm25_by_id.values(), default=0.0)
        now = datetime.now(UTC)
        matches = [
            _memory_match(
                memory,
                exact_key=False,
                bm25_score=bm25_by_id[memory.memory_id],
                maximum_bm25=maximum_bm25,
                now=now,
            )
            for memory, _ in rows
        ]
        matches.sort(
            key=lambda item: (item.relevance, item.memory_id),
            reverse=True,
        )
        return matches[:limit]

    async def _retrieve_memory_vector_scope(
        self,
        *,
        scope: Literal["workspace", "global"],
        user_id: str,
        workspace_id: str,
        candidates: Sequence[tuple[str, float, int]],
        limit: int,
    ) -> list[MemoryMatch]:
        latest_by_id: dict[str, tuple[float, int]] = {}
        for memory_id, similarity, vector_version in candidates:
            if not memory_id or vector_version < 1:
                continue
            previous = latest_by_id.get(memory_id)
            if previous is None or similarity > previous[0]:
                latest_by_id[memory_id] = (similarity, vector_version)
        if not latest_by_id:
            return []
        filters = [
            AgentMemory.memory_id.in_(latest_by_id),
            AgentMemory.scope == scope,
            AgentMemory.user_id == user_id,
            AgentMemory.status == "active",
        ]
        if scope == "workspace":
            filters.append(AgentMemory.workspace_id == workspace_id)
        else:
            filters.append(AgentMemory.workspace_id.is_(None))
        async with self._database.session() as session:
            memories = list(await session.scalars(select(AgentMemory).where(*filters)))
        now = datetime.now(UTC)
        matches = [
            _memory_match(
                memory,
                exact_key=False,
                bm25_score=0.0,
                maximum_bm25=0.0,
                vector_similarity=latest_by_id[memory.memory_id][0],
                now=now,
            )
            for memory in memories
            if memory.version == latest_by_id[memory.memory_id][1]
        ]
        matches.sort(key=lambda item: (item.relevance, item.memory_id), reverse=True)
        return matches[:limit]

    async def mark_outbox_published(self, *, event_id: str) -> None:
        async with self._database.session() as session, session.begin():
            event = await session.get(AgentMemoryOutbox, event_id, with_for_update=True)
            if event is None or event.status != "pending":
                return
            event.status = "published"
            event.attempts += 1
            event.published_at = datetime.now(UTC)

    async def mark_vector_outbox_published(self, *, event_id: str) -> None:
        async with self._database.session() as session, session.begin():
            event = await session.get(AgentMemoryVectorOutbox, event_id, with_for_update=True)
            if event is None or event.status != "pending":
                return
            event.status = "published"
            event.attempts += 1
            event.published_at = datetime.now(UTC)

    async def claim_vector_index(
        self,
        *,
        event: MemoryVectorOutboxEvent,
        worker_id: str,
        lease_seconds: float,
    ) -> MemoryVectorIndexClaim:
        """Serialize derived-index writes for one memory without blocking mutations."""

        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        async with self._database.session() as session, session.begin():
            outbox = await session.get(
                AgentMemoryVectorOutbox,
                event.event_id,
                with_for_update=True,
            )
            if outbox is None or outbox.status == "completed":
                return MemoryVectorIndexClaim(status="already_completed")
            memory = await session.get(AgentMemory, event.memory_id, with_for_update=True)
            if memory is None:
                return MemoryVectorIndexClaim(
                    status="claimed",
                    document=AgentMemoryVectorDocument(
                        memory_id=event.memory_id,
                        active=False,
                    ),
                )
            if (
                memory.vector_lease_expires_at is not None
                and memory.vector_lease_expires_at > now
                and memory.vector_lease_owner != worker_id
            ):
                return MemoryVectorIndexClaim(status="lease_held")
            memory.vector_lease_owner = worker_id
            memory.vector_lease_expires_at = now + timedelta(seconds=lease_seconds)
            return MemoryVectorIndexClaim(
                status="claimed",
                document=_memory_vector_document(memory),
            )

    async def renew_vector_index_lease(
        self,
        *,
        event: MemoryVectorOutboxEvent,
        worker_id: str,
        lease_seconds: float,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        async with self._database.session() as session, session.begin():
            outbox = await session.get(AgentMemoryVectorOutbox, event.event_id)
            if outbox is None or outbox.status == "completed":
                return False
            memory = await session.get(AgentMemory, event.memory_id, with_for_update=True)
            if memory is None:
                return True
            if memory.vector_lease_owner != worker_id:
                return False
            memory.vector_lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)
            return True

    async def complete_vector_index(
        self,
        *,
        event: MemoryVectorOutboxEvent,
        worker_id: str,
    ) -> None:
        """Mark the outbox event complete only after Milvus accepted current truth."""

        async with self._database.session() as session, session.begin():
            outbox = await session.get(
                AgentMemoryVectorOutbox,
                event.event_id,
                with_for_update=True,
            )
            if outbox is None or outbox.status == "completed":
                return
            memory = await session.get(AgentMemory, event.memory_id, with_for_update=True)
            if memory is not None:
                if memory.vector_lease_owner != worker_id:
                    raise RuntimeError("memory vector lease is no longer owned by worker")
                memory.vector_lease_owner = None
                memory.vector_lease_expires_at = None
            outbox.status = "completed"
            outbox.completed_at = datetime.now(UTC)

    async def release_outbox_for_retry(
        self,
        *,
        event_id: str,
        delay_seconds: float,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must not be negative")
        async with self._database.session() as session, session.begin():
            event = await session.get(AgentMemoryOutbox, event_id, with_for_update=True)
            if event is None:
                return
            event.attempts += 1
            event.available_at = datetime.now(UTC) + timedelta(seconds=delay_seconds)

    async def claim_consolidation(
        self,
        *,
        event: MemoryOutboxEvent,
        worker_id: str,
        lease_seconds: float,
    ) -> MemoryConsolidationClaim:
        """Fence one thread so distinct workers never consolidate it concurrently."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(UTC)
        async with self._database.session() as session, session.begin():
            outbox = await session.get(
                AgentMemoryOutbox,
                event.event_id,
                with_for_update=True,
            )
            if outbox is None:
                return MemoryConsolidationClaim(status="already_consolidated")
            thread = await self._locked_thread(
                session,
                thread_id=event.thread_id,
                user_id=event.user_id,
                workspace_id=event.workspace_id,
                create=False,
            )
            self._require_not_deleted(thread)
            if thread.last_consolidated_sequence >= event.through_sequence:
                return MemoryConsolidationClaim(status="already_consolidated")
            if (
                thread.memory_lease_expires_at is not None
                and thread.memory_lease_expires_at > now
                and thread.memory_lease_owner != worker_id
            ):
                return MemoryConsolidationClaim(status="lease_held")
            thread.memory_lease_owner = worker_id
            thread.memory_lease_expires_at = now + timedelta(seconds=lease_seconds)
            messages = list(
                await session.scalars(
                    select(AgentMessage)
                    .where(
                        AgentMessage.thread_id == event.thread_id,
                        AgentMessage.sequence > thread.last_consolidated_sequence,
                        AgentMessage.sequence <= event.through_sequence,
                    )
                    .order_by(AgentMessage.sequence)
                )
            )
            profile = await session.get(WorkspaceMemoryProfile, event.workspace_id)
            active_topics = list(profile.active_topics) if profile is not None else []
            return MemoryConsolidationClaim(
                status="claimed",
                consolidation=ClaimedMemoryConsolidation(
                    event=event,
                    messages=[_message_record(item) for item in messages],
                    thread=_thread_record(thread),
                    active_topics=active_topics,
                ),
            )

    async def renew_consolidation_lease(
        self,
        *,
        event: MemoryOutboxEvent,
        worker_id: str,
        lease_seconds: float,
    ) -> bool:
        """Extend the fence while model calls run; false means this Worker lost it."""

        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=event.thread_id,
                user_id=event.user_id,
                workspace_id=event.workspace_id,
                create=False,
            )
            if (
                thread.last_consolidated_sequence >= event.through_sequence
                or thread.memory_lease_owner != worker_id
            ):
                return False
            thread.memory_lease_expires_at = datetime.now(UTC) + timedelta(
                seconds=lease_seconds
            )
            return True

    async def persist_short_term_consolidation(
        self,
        *,
        claimed: ClaimedMemoryConsolidation,
        worker_id: str,
        summary: ConversationSummary,
        max_active_topics: int,
    ) -> ClaimedMemoryConsolidation:
        """Commit the summary and topic profile before memory extraction begins."""

        if max_active_topics < 1:
            raise ValueError("max_active_topics must be positive")
        if summary.covered_sequence != claimed.event.through_sequence:
            raise ValueError("summary must cover exactly the claimed message range")
        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=claimed.event.thread_id,
                user_id=claimed.event.user_id,
                workspace_id=claimed.event.workspace_id,
                create=False,
            )
            self._require_not_deleted(thread)
            if thread.memory_lease_owner != worker_id:
                raise RuntimeError("memory consolidation lease is no longer owned by worker")
            if thread.last_consolidated_sequence >= claimed.event.through_sequence:
                raise RuntimeError("memory consolidation is already complete")
            if thread.summary_covered_sequence > claimed.event.through_sequence:
                raise RuntimeError("summary cursor is ahead of the claimed message range")
            if thread.summary_covered_sequence < claimed.event.through_sequence:
                thread.summary = summary.summary
                thread.summary_topic = summary.topic
                thread.summary_covered_sequence = summary.covered_sequence
                active_topics = await self._refresh_profile(
                    session,
                    workspace_id=thread.workspace_id,
                    topic=summary.topic,
                    max_active_topics=max_active_topics,
                )
                thread.memory_revision += 1
            else:
                active_topics = await self._profile_topics(
                    session,
                    workspace_id=thread.workspace_id,
                )
            return replace(
                claimed,
                thread=_thread_record(thread),
                active_topics=active_topics,
            )

    async def complete_consolidation(
        self,
        *,
        claimed: ClaimedMemoryConsolidation,
        worker_id: str,
        consolidation: MemoryConsolidation,
    ) -> None:
        """Write memory mutations and then advance the long-memory cursor."""

        async with self._database.session() as session, session.begin():
            thread = await self._locked_thread(
                session,
                thread_id=claimed.event.thread_id,
                user_id=claimed.event.user_id,
                workspace_id=claimed.event.workspace_id,
                create=False,
            )
            self._require_not_deleted(thread)
            if thread.memory_lease_owner != worker_id:
                raise RuntimeError("memory consolidation lease is no longer owned by worker")
            if thread.last_consolidated_sequence >= claimed.event.through_sequence:
                return
            if consolidation.summary.covered_sequence != claimed.event.through_sequence:
                raise ValueError("summary must cover exactly the claimed message range")
            if thread.summary_covered_sequence != claimed.event.through_sequence:
                raise RuntimeError("short-term summary must be committed before memory writes")
            changed_memories = await self._apply_mutations(
                session,
                thread=thread,
                source_messages=claimed.messages,
                mutations=consolidation.mutations,
            )
            for memory in {item.memory_id: item for item in changed_memories}.values():
                session.add(
                    AgentMemoryVectorOutbox(
                        memory_id=memory.memory_id,
                        memory_version=memory.version,
                    )
                )
            thread.last_consolidated_sequence = claimed.event.through_sequence
            thread.memory_lease_owner = None
            thread.memory_lease_expires_at = None

    async def _profile_topics(
        self,
        session: AsyncSession,
        *,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        profile = await session.get(
            WorkspaceMemoryProfile,
            workspace_id,
            with_for_update=True,
        )
        return list(profile.active_topics) if profile is not None else []

    async def _apply_mutations(
        self,
        session: AsyncSession,
        *,
        thread: AgentThread,
        source_messages: Sequence[AgentMessageRecord],
        mutations: Sequence[MemoryMutation],
    ) -> list[AgentMemory]:
        changed: list[AgentMemory] = []
        for mutation in mutations:
            memory = await self._apply_mutation(session, thread=thread, mutation=mutation)
            if memory is None:
                continue
            changed.append(memory)
            for message in source_messages:
                session.add(
                    AgentMemorySource(
                        memory_id=memory.memory_id,
                        thread_id=thread.thread_id,
                        message_id=message.message_id,
                        relation=mutation.action,
                        confidence_delta=mutation.confidence_delta,
                    )
                )
        return changed

    async def _apply_mutation(
        self,
        session: AsyncSession,
        *,
        thread: AgentThread,
        mutation: MemoryMutation,
    ) -> AgentMemory | None:
        now = datetime.now(UTC)
        if mutation.action == "create":
            candidate = mutation.candidate
            if candidate is None:  # Pydantic contract protects this branch.
                raise ValueError("create mutation requires candidate")
            if candidate.scope == "workspace":
                workspace_id: str | None = thread.workspace_id
            else:
                workspace_id = None
            new_memory = AgentMemory(
                scope=candidate.scope,
                user_id=thread.user_id,
                workspace_id=workspace_id,
                kind=candidate.kind,
                memory_key=candidate.memory_key,
                value=candidate.value,
                display_text=candidate.display_text,
                topics=candidate.topics,
                initial_confidence=candidate.initial_confidence,
                confidence=candidate.initial_confidence,
                decay_rate=candidate.decay_rate,
                last_reinforced_at=now,
            )
            session.add(new_memory)
            await session.flush()
            return new_memory

        target_id = mutation.target_memory_id
        if target_id is None:  # Pydantic contract protects this branch.
            raise ValueError("mutation requires target_memory_id")
        target_memory = await session.get(AgentMemory, target_id, with_for_update=True)
        if target_memory is None:
            raise ValueError(f"memory does not exist: {target_id}")
        _validate_memory_owner(target_memory, thread=thread)
        if mutation.action == "merge":
            candidate = mutation.candidate
            if candidate is None:
                raise ValueError("merge mutation requires candidate")
            target_memory.value = candidate.value
            target_memory.display_text = candidate.display_text
            target_memory.topics = candidate.topics
            target_memory.decay_rate = candidate.decay_rate
            target_memory.confidence = min(
                1.0,
                target_memory.confidence + max(0.0, mutation.confidence_delta),
            )
            target_memory.last_reinforced_at = now
            target_memory.version += 1
        elif mutation.action == "deactivate":
            target_memory.status = "inactive"
            target_memory.version += 1
        elif mutation.action == "lower_confidence":
            target_memory.confidence = max(
                0.0,
                target_memory.confidence + mutation.confidence_delta,
            )
            target_memory.version += 1
        else:  # pragma: no cover - model validation bounds the Literal.
            raise ValueError(f"unsupported memory mutation: {mutation.action}")
        return target_memory

    async def _refresh_profile(
        self,
        session: AsyncSession,
        *,
        workspace_id: str,
        topic: str | None,
        max_active_topics: int,
    ) -> list[dict[str, Any]]:
        profile = await session.get(WorkspaceMemoryProfile, workspace_id, with_for_update=True)
        if topic is None:
            return list(profile.active_topics) if profile is not None else []
        if profile is None:
            profile = WorkspaceMemoryProfile(workspace_id=workspace_id)
            session.add(profile)
            await session.flush()
        now = datetime.now(UTC)
        updated = False
        topics = [dict(item) for item in profile.active_topics]
        for item in topics:
            if item.get("topic") == topic:
                item["last_active_at"] = now.isoformat()
                item["thread_count"] = int(item.get("thread_count", 0)) + 1
                updated = True
                break
        if not updated:
            topics.append(
                {"topic": topic, "last_active_at": now.isoformat(), "thread_count": 1}
            )
        topics.sort(
            key=lambda item: (
                str(item.get("last_active_at", "")),
                int(item.get("thread_count", 0)),
            ),
            reverse=True,
        )
        profile.active_topics = topics[:max_active_topics]
        profile.primary_topic = str(profile.active_topics[0]["topic"])
        profile.revision += 1
        return [dict(item) for item in profile.active_topics]

    async def _locked_thread(
        self,
        session: AsyncSession,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
        create: bool,
    ) -> AgentThread:
        thread = await session.get(AgentThread, thread_id, with_for_update=True)
        if thread is None:
            if not create:
                raise ValueError(f"agent thread does not exist: {thread_id}")
            workspace = await session.get(Workspace, workspace_id)
            if workspace is None:
                raise ValueError(f"workspace does not exist: {workspace_id}")
            thread = AgentThread(
                thread_id=thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
            session.add(thread)
            await session.flush()
            return thread
        if thread.user_id != user_id or thread.workspace_id != workspace_id:
            raise PermissionError("agent thread does not belong to this user and workspace")
        return thread

    async def _thread_for_identity(
        self,
        session: AsyncSession,
        *,
        thread_id: str,
        user_id: str,
        workspace_id: str,
    ) -> AgentThread:
        thread = await session.get(AgentThread, thread_id)
        if thread is None:
            raise ValueError(f"agent thread does not exist: {thread_id}")
        if thread.user_id != user_id or thread.workspace_id != workspace_id:
            raise PermissionError("agent thread does not belong to this user and workspace")
        return thread

    @staticmethod
    def _require_active(thread: AgentThread) -> None:
        if thread.status != "active":
            raise AgentThreadStateError(
                f"agent thread is {thread.status} and cannot accept new messages"
            )

    @staticmethod
    def _require_not_deleted(thread: AgentThread) -> None:
        if thread.status == "deleted":
            raise ValueError("agent thread has been deleted")


def estimate_message_tokens(content: Any) -> int:
    """Cheap, deterministic fallback used only to decide deferred summarization."""

    encoded = json.dumps(content, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, (len(encoded) + 3) // 4)


def _message_record(message: AgentMessage) -> AgentMessageRecord:
    return AgentMessageRecord(
        message_id=message.message_id,
        thread_id=message.thread_id,
        sequence=message.sequence,
        role=message.role,
        content=message.content,
        name=message.name,
        turn_id=message.turn_id,
        request_id=message.request_id,
        estimated_tokens=message.estimated_tokens,
        created_at=message.created_at,
    )


def _thread_record(thread: AgentThread) -> AgentThreadRecord:
    return AgentThreadRecord(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        workspace_id=thread.workspace_id,
        title=thread.title,
        status=thread.status,
        summary=thread.summary,
        summary_topic=thread.summary_topic,
        summary_covered_sequence=thread.summary_covered_sequence,
        last_consolidated_sequence=thread.last_consolidated_sequence,
        memory_revision=thread.memory_revision,
        last_message_at=thread.last_message_at,
        last_message_sequence=thread.last_message_sequence,
        deleted_at=thread.deleted_at,
    )


def _outbox_event(event: AgentMemoryOutbox) -> MemoryOutboxEvent:
    return MemoryOutboxEvent(
        event_id=event.event_id,
        thread_id=event.thread_id,
        user_id=event.user_id,
        workspace_id=event.workspace_id,
        through_sequence=event.through_sequence,
    )


def _vector_outbox_event(event: AgentMemoryVectorOutbox) -> MemoryVectorOutboxEvent:
    return MemoryVectorOutboxEvent(
        event_id=event.event_id,
        memory_id=event.memory_id,
        memory_version=event.memory_version,
    )


def _memory_vector_document(memory: AgentMemory) -> AgentMemoryVectorDocument:
    return AgentMemoryVectorDocument(
        memory_id=memory.memory_id,
        active=memory.status == "active",
        scope=memory.scope,
        user_id=memory.user_id,
        workspace_id=memory.workspace_id,
        kind=memory.kind,
        memory_key=memory.memory_key,
        display_text=memory.display_text,
        topics=list(memory.topics),
        version=memory.version,
    )


def _validate_memory_owner(memory: AgentMemory, *, thread: AgentThread) -> None:
    if memory.user_id != thread.user_id:
        raise PermissionError("memory does not belong to this user")
    if memory.scope == "workspace" and memory.workspace_id != thread.workspace_id:
        raise PermissionError("workspace memory does not belong to this workspace")


def _memory_match(
    memory: AgentMemory,
    *,
    exact_key: bool,
    bm25_score: float,
    maximum_bm25: float,
    vector_similarity: float = 0.0,
    now: datetime,
) -> MemoryMatch:
    age_days = max(0.0, (now - memory.last_reinforced_at).total_seconds() / 86_400)
    effective_confidence = memory.confidence * math.exp(-memory.decay_rate * age_days)
    normalized_bm25 = bm25_score / maximum_bm25 if maximum_bm25 > 0 else 0.0
    normalized_vector = min(1.0, max(0.0, vector_similarity))
    relevance = max(1.0 if exact_key else 0.0, normalized_bm25, normalized_vector)
    relevance *= effective_confidence
    return MemoryMatch(
        memory_id=memory.memory_id,
        scope=memory.scope,
        kind=memory.kind,
        memory_key=memory.memory_key,
        value=memory.value,
        display_text=memory.display_text,
        topics=memory.topics,
        confidence=memory.confidence,
        effective_confidence=effective_confidence,
        decay_rate=memory.decay_rate,
        status=memory.status,
        relevance=relevance,
    )


def _agent_memory_context(match: MemoryMatch) -> dict[str, Any]:
    return {
        "memory_id": match.memory_id,
        "scope": match.scope,
        "kind": match.kind,
        "key": match.memory_key,
        "value": match.value,
        "text": match.display_text,
        "topics": match.topics,
        "confidence": match.effective_confidence,
    }


class PostgresAgentMemoryStore:
    """Read structured memory for graph context; the Worker owns all writes."""

    def __init__(
        self,
        repository: AgentConversationRepository,
        *,
        per_scope_limit: int,
    ) -> None:
        self._repository = repository
        self._per_scope_limit = per_scope_limit

    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return await self._repository.retrieve_memory_context(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
            per_scope_limit=self._per_scope_limit,
        )

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None:
        """Ignore legacy graph-side writes; deferred consolidation is authoritative."""

        del user_id, workspace_id, writes
