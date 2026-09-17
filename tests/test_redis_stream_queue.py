"""Contracts shared by the material and Agent-memory Redis transports."""

from __future__ import annotations

from typing import Any

import pytest

from capsule.agent.memory_worker import MemoryQueueMessage, RedisMemoryQueue
from capsule.pipeline.redis_stream_queue import RedisStreamQueue


class _FakeRedis:
    def __init__(self) -> None:
        self.created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.added: list[tuple[str, dict[str, str]]] = []
        self.claimed: list[tuple[str, dict[str, str]]] = []
        self.new_messages: list[tuple[str, dict[str, str]]] = []
        self.acked: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.closed = False

    async def xgroup_create(self, *args: Any, **kwargs: Any) -> None:
        self.created.append((args, kwargs))

    async def xadd(self, stream: str, fields: dict[str, str]) -> str:
        self.added.append((stream, fields))
        return "12-0"

    async def xautoclaim(self, *_args: Any, **_kwargs: Any) -> tuple[str, list[Any], list[Any]]:
        claimed, self.claimed = self.claimed, []
        return ("0-0", claimed, [])

    async def xreadgroup(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        messages, self.new_messages = self.new_messages, []
        return [("stream", messages)] if messages else []

    async def xack(self, stream: str, group: str, receipt: str) -> int:
        self.acked.append((stream, group, receipt))
        return 1

    async def xdel(self, stream: str, receipt: str) -> int:
        self.deleted.append((stream, receipt))
        return 1

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_shared_transport_recovers_pending_entries_and_keeps_stream_history() -> None:
    client = _FakeRedis()
    client.claimed = [("9-0", {"value": "recovered"})]
    queue = RedisStreamQueue[str](
        redis_url="redis://unused",
        stream="capsule:test:shared",
        group="workers",
        consumer="worker-a",
        encode=lambda message: {"value": message},
        decode=lambda fields: str(fields["value"]),
        client=client,
    )

    await queue.start()
    assert await queue.publish("new") == "12-0"
    delivery = await queue.receive()
    await queue.acknowledge(delivery)
    await queue.close()

    assert client.created == [(("capsule:test:shared", "workers"), {"id": "0-0", "mkstream": True})]
    assert client.added == [("capsule:test:shared", {"value": "new"})]
    assert delivery.receipt == "9-0"
    assert delivery.message == "recovered"
    assert client.acked == [("capsule:test:shared", "workers", "9-0")]
    assert client.deleted == []
    assert client.closed


@pytest.mark.asyncio
async def test_memory_adapter_uses_shared_transport_but_deletes_acknowledged_events() -> None:
    client = _FakeRedis()
    client.new_messages = [
        (
            "10-0",
            {
                "event_id": "event-1",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "workspace_id": "workspace-1",
                "through_sequence": "9",
            },
        )
    ]
    queue = RedisMemoryQueue(
        redis_url="redis://unused",
        stream="capsule:agent-memory",
        group="memory-workers",
        consumer="worker-a",
        client=client,
    )

    await queue.start()
    delivery = await queue.receive()
    await queue.acknowledge(delivery)

    assert delivery.message == MemoryQueueMessage(
        event_id="event-1",
        thread_id="thread-1",
        user_id="user-1",
        workspace_id="workspace-1",
        through_sequence=9,
    )
    assert client.acked == [("capsule:agent-memory", "memory-workers", "10-0")]
    assert client.deleted == [("capsule:agent-memory", "10-0")]
