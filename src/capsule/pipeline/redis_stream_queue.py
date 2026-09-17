"""Domain-neutral Redis Streams transport for durable task workers.

The transport owns consumer-group setup, pending-entry recovery and acknowledgement.
Each domain owns its message contract, durable state and failure semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

MessageT = TypeVar("MessageT")
StreamFields = Mapping[str, str | bytes]
StreamEncoder = Callable[[MessageT], Mapping[str, str]]
StreamDecoder = Callable[[StreamFields], MessageT]


@dataclass(frozen=True, slots=True)
class RedisStreamDelivery(Generic[MessageT]):
    """One claimed consumer-group entry and its decoded domain message."""

    message: MessageT
    receipt: str


class RedisStreamQueue(Generic[MessageT]):
    """Reusable Redis Streams delivery transport with PEL recovery.

    The queue intentionally has no task retry or dead-letter policy. Those
    policies need the durable state model of the domain that owns the task.
    """

    def __init__(
        self,
        *,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        encode: StreamEncoder[MessageT],
        decode: StreamDecoder[MessageT],
        claim_idle_ms: int = 30_000,
        delete_on_ack: bool = False,
        client: Any | None = None,
    ) -> None:
        if claim_idle_ms < 1:
            raise ValueError("claim_idle_ms must be positive")
        self._redis_url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._encode = encode
        self._decode = decode
        self._claim_idle_ms = claim_idle_ms
        self._delete_on_ack = delete_on_ack
        self._client = client

    async def start(self) -> None:
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._redis_url, decode_responses=True)
        try:
            await self._client.xgroup_create(self._stream, self._group, id="0-0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def publish(self, message: MessageT) -> str:
        return str(await self._required_client().xadd(self._stream, self._encode(message)))

    async def receive(self) -> RedisStreamDelivery[MessageT]:
        client = self._required_client()
        while True:
            await self._before_receive()
            claimed = await client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._claim_idle_ms,
                start_id="0-0",
                count=1,
            )
            messages = claimed[1] if len(claimed) > 1 else []
            if messages:
                delivery = await self._decode_or_handle(messages[0])
                if delivery is not None:
                    return delivery
                continue
            response = await client.xreadgroup(
                self._group,
                self._consumer,
                {self._stream: ">"},
                count=1,
                block=1_000,
            )
            if response:
                delivery = await self._decode_or_handle(response[0][1][0])
                if delivery is not None:
                    return delivery

    async def acknowledge(self, delivery: RedisStreamDelivery[MessageT]) -> None:
        await self._acknowledge_receipt(delivery.receipt)
        if self._delete_on_ack:
            await self._required_client().xdel(self._stream, delivery.receipt)

    async def _before_receive(self) -> None:
        """Allow domains with delayed retries to release due entries first."""

    async def _decode_or_handle(
        self,
        message: tuple[str | bytes, StreamFields],
    ) -> RedisStreamDelivery[MessageT] | None:
        receipt, fields = message
        normalized_receipt = receipt.decode() if isinstance(receipt, bytes) else str(receipt)
        try:
            return RedisStreamDelivery(
                message=self._decode(fields),
                receipt=normalized_receipt,
            )
        except (TypeError, ValueError) as exc:
            return await self._handle_decode_error(
                receipt=normalized_receipt,
                fields=fields,
                error=exc,
            )

    async def _handle_decode_error(
        self,
        *,
        receipt: str,
        fields: StreamFields,
        error: TypeError | ValueError,
    ) -> RedisStreamDelivery[MessageT] | None:
        """Raise by default so domains cannot silently discard bad entries."""

        del receipt, fields
        raise error

    async def _acknowledge_receipt(self, receipt: str) -> None:
        await self._required_client().xack(self._stream, self._group, receipt)

    def _required_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("Redis stream queue has not been started")
        return self._client
