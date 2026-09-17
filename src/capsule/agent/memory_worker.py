"""Durable, non-agent execution flow for deferred conversation consolidation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from capsule.agent.memory_contracts import MemoryConsolidation
from capsule.db.agent_memory import (
    AgentConversationRepository,
    ClaimedMemoryConsolidation,
    MemoryOutboxEvent,
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
    def from_fields(cls, fields: dict[str, str]) -> MemoryQueueMessage:
        try:
            return cls(
                event_id=fields["event_id"],
                thread_id=fields["thread_id"],
                user_id=fields["user_id"],
                workspace_id=fields["workspace_id"],
                through_sequence=int(fields["through_sequence"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid agent-memory queue message") from exc


@dataclass(frozen=True, slots=True)
class MemoryQueueDelivery:
    message: MemoryQueueMessage
    receipt: str


class MemoryQueue(Protocol):
    async def start(self) -> None: ...

    async def publish(self, message: MemoryQueueMessage) -> None: ...

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

    async def publish(self, message: MemoryQueueMessage) -> None:
        self._sequence += 1
        await self._items.put(
            MemoryQueueDelivery(message=message, receipt=f"memory-{self._sequence}")
        )

    async def receive(self) -> MemoryQueueDelivery:
        return await self._items.get()

    async def acknowledge(self, delivery: MemoryQueueDelivery) -> None:
        del delivery
        self._items.task_done()

    async def close(self) -> None:
        return None


class RedisMemoryQueue:
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
        self._redis_url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._claim_idle_ms = claim_idle_ms
        self._client = client

    async def start(self) -> None:
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._redis_url, decode_responses=True)
        try:
            await self._client.xgroup_create(
                self._stream,
                self._group,
                id="0-0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def publish(self, message: MemoryQueueMessage) -> None:
        await self._required_client().xadd(self._stream, message.to_fields())

    async def receive(self) -> MemoryQueueDelivery:
        client = self._required_client()
        while True:
            claimed = await client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._claim_idle_ms,
                start_id="0-0",
                count=1,
            )
            claimed_messages = claimed[1] if len(claimed) > 1 else []
            if claimed_messages:
                return _redis_delivery(claimed_messages[0])
            response = await client.xreadgroup(
                self._group,
                self._consumer,
                {self._stream: ">"},
                count=1,
                block=1_000,
            )
            if response:
                return _redis_delivery(response[0][1][0])

    async def acknowledge(self, delivery: MemoryQueueDelivery) -> None:
        client = self._required_client()
        await client.xack(self._stream, self._group, delivery.receipt)
        await client.xdel(self._stream, delivery.receipt)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _required_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Redis memory queue has not been started")
        return self._client


def _redis_delivery(message: tuple[str, dict[str, str]]) -> MemoryQueueDelivery:
    receipt, fields = message
    decoded = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in fields.items()
    }
    return MemoryQueueDelivery(
        message=MemoryQueueMessage.from_fields(decoded),
        receipt=str(receipt),
    )


class MemoryConsolidator(Protocol):
    """Model-backed extraction node used by the deterministic worker flow."""

    async def consolidate(
        self,
        claimed: ClaimedMemoryConsolidation,
    ) -> MemoryConsolidation: ...


class MemoryOutboxDispatcher:
    """Publish committed events; Postgres remains authoritative when Redis is down."""

    def __init__(self, *, repository: AgentConversationRepository, queue: MemoryQueue) -> None:
        self._repository = repository
        self._queue = queue

    async def dispatch_once(self, *, limit: int = 100) -> int:
        published = 0
        for event in await self._repository.pending_outbox_events(limit=limit):
            await self._queue.publish(MemoryQueueMessage.from_event(event))
            await self._repository.mark_outbox_published(event_id=event.event_id)
            published += 1
        return published


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
            consolidation = await self._consolidator.consolidate(claimed)
            await self._repository.complete_consolidation(
                claimed=claimed,
                worker_id=self._worker_id,
                consolidation=consolidation,
                max_active_topics=self._max_active_topics,
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


async def run_memory_worker(
    worker: MemoryWorker,
    *,
    stop_requested: Callable[[], bool],
) -> None:
    """Run a long-lived worker without imposing a host-specific process manager."""

    while not stop_requested():
        await worker.run_once()
