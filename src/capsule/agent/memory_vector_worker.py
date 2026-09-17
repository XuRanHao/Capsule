"""Durably project structured Agent memories into the isolated Milvus collection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from capsule.agent.milvus_memory_store import TextEmbedder
from capsule.db.agent_memory import (
    AgentConversationRepository,
    AgentMemoryVectorDocument,
    MemoryVectorOutboxEvent,
)
from capsule.pipeline.redis_stream_queue import (
    RedisStreamDelivery,
    RedisStreamQueue,
    StreamFields,
)
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore, AgentMemoryVectorRecord


@dataclass(frozen=True, slots=True)
class MemoryVectorQueueMessage:
    event_id: str
    memory_id: str
    memory_version: int

    @classmethod
    def from_event(cls, event: MemoryVectorOutboxEvent) -> MemoryVectorQueueMessage:
        return cls(
            event_id=event.event_id,
            memory_id=event.memory_id,
            memory_version=event.memory_version,
        )

    def to_event(self) -> MemoryVectorOutboxEvent:
        return MemoryVectorOutboxEvent(
            event_id=self.event_id,
            memory_id=self.memory_id,
            memory_version=self.memory_version,
        )

    def to_fields(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "memory_id": self.memory_id,
            "memory_version": str(self.memory_version),
        }

    @classmethod
    def from_fields(cls, fields: StreamFields) -> MemoryVectorQueueMessage:
        decoded = {
            (key.decode() if isinstance(key, bytes) else str(key)): (
                value.decode() if isinstance(value, bytes) else str(value)
            )
            for key, value in fields.items()
        }
        try:
            return cls(
                event_id=decoded["event_id"],
                memory_id=decoded["memory_id"],
                memory_version=int(decoded["memory_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid agent-memory-vector queue message") from exc


@dataclass(frozen=True, slots=True)
class MemoryVectorQueueDelivery(RedisStreamDelivery[MemoryVectorQueueMessage]):
    """The vector worker's domain-specific Stream delivery."""


class MemoryVectorQueue(Protocol):
    async def start(self) -> None: ...

    async def publish(self, message: MemoryVectorQueueMessage) -> str: ...

    async def receive(self) -> MemoryVectorQueueDelivery: ...

    async def acknowledge(self, delivery: MemoryVectorQueueDelivery) -> None: ...

    async def close(self) -> None: ...


class RedisMemoryVectorQueue(RedisStreamQueue[MemoryVectorQueueMessage]):
    """The same generic Redis Stream transport used by materials and summaries."""

    def __init__(
        self,
        *,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        claim_idle_ms: int = 60_000,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            redis_url=redis_url,
            stream=stream,
            group=group,
            consumer=consumer,
            encode=lambda message: message.to_fields(),
            decode=MemoryVectorQueueMessage.from_fields,
            claim_idle_ms=claim_idle_ms,
            delete_on_ack=True,
            client=client,
        )

    async def receive(self) -> MemoryVectorQueueDelivery:
        delivery = await super().receive()
        return MemoryVectorQueueDelivery(message=delivery.message, receipt=delivery.receipt)


class MemoryVectorOutboxDispatcher:
    """Bridge committed vector-index requests from PostgreSQL into Redis Streams."""

    def __init__(
        self,
        *,
        repository: AgentConversationRepository,
        queue: MemoryVectorQueue,
    ) -> None:
        self._repository = repository
        self._queue = queue

    async def dispatch_once(self, *, limit: int = 100) -> int:
        published = 0
        for event in await self._repository.pending_vector_outbox_events(limit=limit):
            await self._queue.publish(MemoryVectorQueueMessage.from_event(event))
            await self._repository.mark_vector_outbox_published(event_id=event.event_id)
            published += 1
        return published


class MemoryVectorWorker:
    """Create or remove derived vectors without making Milvus a source of truth."""

    def __init__(
        self,
        *,
        worker_id: str,
        repository: AgentConversationRepository,
        queue: MemoryVectorQueue,
        embedder: TextEmbedder,
        vector_store: AgentMemoryMilvusStore,
        lease_seconds: float,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._worker_id = worker_id
        self._repository = repository
        self._queue = queue
        self._embedder = embedder
        self._vector_store = vector_store
        self._lease_seconds = lease_seconds
        self._collection_ready = False

    async def run_once(self) -> bool:
        delivery = await self._queue.receive()
        event = delivery.message.to_event()
        claim = await self._repository.claim_vector_index(
            event=event,
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if claim.status == "already_completed":
            await self._queue.acknowledge(delivery)
            return False
        if claim.status == "lease_held":
            # The current owner may be embedding or flushing.  Leave this in
            # the pending-entry list so normal Redis recovery can redeliver it.
            return False
        document = claim.document
        if document is None:  # pragma: no cover - repository contract guarantees it.
            raise RuntimeError("claimed memory vector event has no document")
        heartbeat = asyncio.create_task(self._renew_lease_until_complete(event))
        try:
            await self._synchronize(document)
            await self._repository.complete_vector_index(
                event=event,
                worker_id=self._worker_id,
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        await self._queue.acknowledge(delivery)
        return True

    async def _synchronize(self, document: AgentMemoryVectorDocument) -> None:
        if not self._collection_ready:
            await self._vector_store.ensure_collection()
            self._collection_ready = True
        if not document.active:
            await self._vector_store.adelete([document.memory_id])
            return
        vector = (await self._embedder.embed_text(_embedding_text(document))).vector
        if (
            document.scope not in {"workspace", "global"}
            or not document.user_id
            or not document.kind
            or document.version is None
        ):
            raise ValueError("active memory vector document is incomplete")
        await self._vector_store.aupsert(
            [
                AgentMemoryVectorRecord(
                    memory_id=document.memory_id,
                    scope=(
                        "workspace" if document.scope == "workspace" else "global"
                    ),
                    user_id=document.user_id,
                    workspace_id=document.workspace_id,
                    kind=document.kind,
                    memory_version=document.version,
                    vector=vector,
                )
            ]
        )

    async def _renew_lease_until_complete(self, event: MemoryVectorOutboxEvent) -> None:
        interval_seconds = max(0.1, min(30.0, self._lease_seconds / 3))
        while True:
            await asyncio.sleep(interval_seconds)
            renewed = await self._repository.renew_vector_index_lease(
                event=event,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                return


def _embedding_text(document: AgentMemoryVectorDocument) -> str:
    """Use structured cues without inserting raw JSON into the vector model."""

    topics = "、".join(document.topics or [])
    lines = [
        f"记忆类型：{document.kind or ''}",
        f"记忆键：{document.memory_key or ''}",
        f"主题：{topics}",
        f"内容：{document.display_text or ''}",
    ]
    return "\n".join(lines)


async def run_memory_vector_worker(
    worker: MemoryVectorWorker,
    *,
    stop_requested: Callable[[], bool],
) -> None:
    """Host-independent runner matching the normal memory Worker's lifecycle."""

    while not stop_requested():
        await worker.run_once()
