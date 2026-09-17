"""Durable, non-agent execution flow for deferred conversation consolidation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from capsule.agent.memory_contracts import ConversationSummary, MemoryConsolidation
from capsule.db.agent_memory import (
    AgentConversationRepository,
    ClaimedMemoryConsolidation,
    MemoryOutboxEvent,
)
from capsule.pipeline.redis_stream_queue import (
    RedisStreamDelivery,
    RedisStreamQueue,
    StreamFields,
)


@dataclass(frozen=True, slots=True)
class MemoryQueueMessage:
    event_id: str
    thread_id: str
    user_id: str
    workspace_id: str
    through_sequence: int

    @classmethod
    def from_event(cls, event: MemoryOutboxEvent) -> MemoryQueueMessage:
        return cls(
            event_id=event.event_id,
            thread_id=event.thread_id,
            user_id=event.user_id,
            workspace_id=event.workspace_id,
            through_sequence=event.through_sequence,
        )

    def to_event(self) -> MemoryOutboxEvent:
        return MemoryOutboxEvent(
            event_id=self.event_id,
            thread_id=self.thread_id,
            user_id=self.user_id,
            workspace_id=self.workspace_id,
            through_sequence=self.through_sequence,
        )

    def to_fields(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "workspace_id": self.workspace_id,
            "through_sequence": str(self.through_sequence),
        }

    @classmethod
    def from_fields(cls, fields: StreamFields) -> MemoryQueueMessage:
        decoded = {
            (key.decode() if isinstance(key, bytes) else str(key)): (
                value.decode() if isinstance(value, bytes) else str(value)
            )
            for key, value in fields.items()
        }
        try:
            return cls(
                event_id=decoded["event_id"],
                thread_id=decoded["thread_id"],
                user_id=decoded["user_id"],
                workspace_id=decoded["workspace_id"],
                through_sequence=int(decoded["through_sequence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid agent-memory queue message") from exc


@dataclass(frozen=True, slots=True)
class MemoryQueueDelivery(RedisStreamDelivery[MemoryQueueMessage]):
    """Memory-specific delivery type used by the consolidation worker."""


class MemoryQueue(Protocol):
    async def start(self) -> None: ...

    async def publish(self, message: MemoryQueueMessage) -> str: ...

    async def receive(self) -> MemoryQueueDelivery: ...

    async def acknowledge(self, delivery: MemoryQueueDelivery) -> None: ...

    async def close(self) -> None: ...


class InMemoryMemoryQueue:
    """Deterministic transport for tests and local execution without Redis."""

    def __init__(self) -> None:
        self._items: asyncio.Queue[MemoryQueueDelivery] = asyncio.Queue()
        self._sequence = 0

    async def start(self) -> None:
        return None

    async def publish(self, message: MemoryQueueMessage) -> str:
        self._sequence += 1
        await self._items.put(
            MemoryQueueDelivery(message=message, receipt=f"memory-{self._sequence}")
        )
        return f"memory-{self._sequence}"

    async def receive(self) -> MemoryQueueDelivery:
        return await self._items.get()

    async def acknowledge(self, delivery: MemoryQueueDelivery) -> None:
        del delivery
        self._items.task_done()

    async def close(self) -> None:
        return None


class RedisMemoryQueue(RedisStreamQueue[MemoryQueueMessage]):
    """Redis Streams transport with consumer-group recovery for memory workers."""

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
            decode=MemoryQueueMessage.from_fields,
            claim_idle_ms=claim_idle_ms,
            delete_on_ack=True,
            client=client,
        )

    async def receive(self) -> MemoryQueueDelivery:
        delivery = await super().receive()
        return MemoryQueueDelivery(message=delivery.message, receipt=delivery.receipt)


class MemoryConsolidator(Protocol):
    """Model-backed extraction node used by the deterministic worker flow."""

    async def summarize(
        self,
        claimed: ClaimedMemoryConsolidation,
    ) -> ConversationSummary: ...

    async def consolidate(
        self,
        claimed: ClaimedMemoryConsolidation,
        *,
        summary: ConversationSummary,
    ) -> MemoryConsolidation: ...


class MemoryOutboxDispatcher:
    """Publish committed events; Postgres remains authoritative when Redis is down."""

    def __init__(self, *, repository: AgentConversationRepository, queue: MemoryQueue) -> None:
        self._repository = repository
        self._queue = queue

    async def dispatch_once(self, *, limit: int = 100) -> int:
        published = 0
        for event in await self._repository.pending_outbox_events(limit=limit):
            await self.dispatch_event(event)
            published += 1
        return published

    async def dispatch_event(self, event: MemoryOutboxEvent) -> None:
        """Publish one just-created event without waiting for the next scan."""

        await self._queue.publish(MemoryQueueMessage.from_event(event))
        await self._repository.mark_outbox_published(event_id=event.event_id)


class MemoryWorker:
    """One queue consumer; safe to run in multiple independent processes."""

    def __init__(
        self,
        *,
        worker_id: str,
        repository: AgentConversationRepository,
        queue: MemoryQueue,
        consolidator: MemoryConsolidator,
        lease_seconds: float,
        max_active_topics: int,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        if lease_seconds <= 0 or max_active_topics < 1:
            raise ValueError("memory worker limits must be positive")
        self._worker_id = worker_id
        self._repository = repository
        self._queue = queue
        self._consolidator = consolidator
        self._lease_seconds = lease_seconds
        self._max_active_topics = max_active_topics

    async def run_once(self) -> bool:
        delivery = await self._queue.receive()
        claim = await self._repository.claim_consolidation(
            event=delivery.message.to_event(),
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if claim.status == "already_consolidated":
            await self._queue.acknowledge(delivery)
            return False
        if claim.status == "lease_held":
            # Do not ACK: once the other Worker's lease expires, Streams can
            # redeliver this event to finish it safely.
            return False
        claimed = claim.consolidation
        if claimed is None:  # pragma: no cover - repository contract guarantees this.
            raise RuntimeError("claimed memory consolidation is missing its payload")
        heartbeat = asyncio.create_task(self._renew_lease_until_complete(claimed))
        try:
            summary = _staged_summary(claimed)
            if summary is None:
                summary = await self._consolidator.summarize(claimed)
            prepared = await self._repository.persist_short_term_consolidation(
                claimed=claimed,
                worker_id=self._worker_id,
                summary=summary,
                max_active_topics=self._max_active_topics,
            )
            consolidation = await self._consolidator.consolidate(
                prepared,
                summary=summary,
            )
            await self._repository.complete_consolidation(
                claimed=prepared,
                worker_id=self._worker_id,
                consolidation=consolidation,
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        await self._queue.acknowledge(delivery)
        return True

    async def _renew_lease_until_complete(
        self,
        claimed: ClaimedMemoryConsolidation,
    ) -> None:
        interval_seconds = max(0.1, min(30.0, self._lease_seconds / 3))
        while True:
            await asyncio.sleep(interval_seconds)
            renewed = await self._repository.renew_consolidation_lease(
                event=claimed.event,
                worker_id=self._worker_id,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                return


def _staged_summary(
    claimed: ClaimedMemoryConsolidation,
) -> ConversationSummary | None:
    thread = claimed.thread
    if (
        thread.summary_covered_sequence != claimed.event.through_sequence
        or thread.summary is None
    ):
        return None
    return ConversationSummary(
        summary=thread.summary,
        topic=thread.summary_topic,
        covered_sequence=thread.summary_covered_sequence,
    )


async def run_memory_worker(
    worker: MemoryWorker,
    *,
    stop_requested: Callable[[], bool],
) -> None:
    """Run a long-lived worker without imposing a host-specific process manager."""

    while not stop_requested():
        await worker.run_once()
