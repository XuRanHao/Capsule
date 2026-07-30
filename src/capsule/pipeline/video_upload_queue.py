"""Durable queue transports for rendered video upload manifests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class UploadQueueItem:
    manifest_path: str
    attempt: int = 0


@dataclass(slots=True, frozen=True)
class UploadQueueDelivery:
    item: UploadQueueItem
    receipt: str


class UploadQueue(Protocol):
    async def start(self) -> None: ...

    async def publish(self, item: UploadQueueItem) -> None: ...

    async def receive(self) -> UploadQueueDelivery: ...

    async def acknowledge(self, delivery: UploadQueueDelivery) -> None: ...

    async def retry(
        self,
        delivery: UploadQueueDelivery,
        *,
        next_attempt: int | None = None,
    ) -> None: ...

    async def close(self) -> None: ...


class InMemoryUploadQueue:
    """Bounded transport used to validate render/upload decoupling locally."""

    def __init__(self, *, maxsize: int) -> None:
        self._queue: asyncio.Queue[UploadQueueItem] = asyncio.Queue(maxsize=maxsize)

    async def start(self) -> None:
        return None

    async def publish(self, item: UploadQueueItem) -> None:
        await self._queue.put(item)

    async def receive(self) -> UploadQueueDelivery:
        item = await self._queue.get()
        return UploadQueueDelivery(item=item, receipt=item.manifest_path)

    async def acknowledge(self, delivery: UploadQueueDelivery) -> None:
        del delivery
        self._queue.task_done()

    async def retry(
        self,
        delivery: UploadQueueDelivery,
        *,
        next_attempt: int | None = None,
    ) -> None:
        self._queue.task_done()
        await self._queue.put(
            UploadQueueItem(
                manifest_path=delivery.item.manifest_path,
                attempt=next_attempt or delivery.item.attempt + 1,
            )
        )

    async def close(self) -> None:
        return None


class RedisStreamUploadQueue:
    """Redis Streams transport with explicit retries and abandoned-task claiming."""

    def __init__(
        self,
        *,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        claim_idle_ms: int,
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

    async def publish(self, item: UploadQueueItem) -> None:
        client = self._required_client()
        await client.xadd(
            self._stream,
            {"manifest_path": item.manifest_path, "attempt": str(item.attempt)},
        )

    async def receive(self) -> UploadQueueDelivery:
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

    async def acknowledge(self, delivery: UploadQueueDelivery) -> None:
        client = self._required_client()
        await client.xack(self._stream, self._group, delivery.receipt)
        await client.xdel(self._stream, delivery.receipt)

    async def retry(
        self,
        delivery: UploadQueueDelivery,
        *,
        next_attempt: int | None = None,
    ) -> None:
        # Re-add + ack makes the attempt count durable. If the process dies between
        # those operations, the original remains pending and XAUTOCLAIM recovers it.
        await self.publish(
            UploadQueueItem(
                manifest_path=delivery.item.manifest_path,
                attempt=next_attempt or delivery.item.attempt + 1,
            )
        )
        await self.acknowledge(delivery)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _required_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Redis upload queue has not been started")
        return self._client


def _redis_delivery(message: tuple[str, dict[str, str]]) -> UploadQueueDelivery:
    receipt, fields = message
    return UploadQueueDelivery(
        item=UploadQueueItem(
            manifest_path=fields["manifest_path"],
            attempt=int(fields.get("attempt", "0")),
        ),
        receipt=receipt,
    )
