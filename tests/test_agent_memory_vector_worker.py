import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from capsule.agent.memory_vector_worker import (
    MemoryVectorOutboxDispatcher,
    MemoryVectorQueueDelivery,
    MemoryVectorQueueMessage,
    MemoryVectorWorker,
)
from capsule.db.agent_memory import (
    AgentMemoryVectorDocument,
    MemoryVectorIndexClaim,
    MemoryVectorOutboxEvent,
)
from capsule.vectorstore.agent_memory import AgentMemoryVectorRecord


def _event() -> MemoryVectorOutboxEvent:
    return MemoryVectorOutboxEvent(
        event_id="event-1",
        memory_id="mem-1",
        memory_version=2,
    )


@dataclass
class _Queue:
    messages: list[MemoryVectorQueueMessage]
    acknowledged: list[str]

    async def publish(self, message: MemoryVectorQueueMessage) -> str:
        self.messages.append(message)
        return f"vector-{len(self.messages)}"

    async def receive(self) -> MemoryVectorQueueDelivery:
        return MemoryVectorQueueDelivery(
            message=self.messages.pop(0),
            receipt="vector-1",
        )

    async def acknowledge(self, delivery: MemoryVectorQueueDelivery) -> None:
        self.acknowledged.append(delivery.receipt)


class _VectorStore:
    def __init__(self) -> None:
        self.ensured = 0
        self.records: list[AgentMemoryVectorRecord] = []
        self.deleted: list[str] = []

    async def ensure_collection(self) -> bool:
        self.ensured += 1
        return True

    async def aupsert(self, records: list[AgentMemoryVectorRecord]) -> None:
        self.records.extend(records)

    async def adelete(self, memory_ids: list[str]) -> int:
        self.deleted.extend(memory_ids)
        return len(memory_ids)


@pytest.mark.asyncio
async def test_vector_outbox_dispatcher_publishes_then_marks_postgres_event() -> None:
    class Repository:
        marked: list[str] = []

        async def pending_vector_outbox_events(self, *, limit: int):
            assert limit == 100
            return [_event()]

        async def mark_vector_outbox_published(self, *, event_id: str) -> None:
            self.marked.append(event_id)

    queue = _Queue(messages=[], acknowledged=[])
    repository = Repository()
    dispatcher = MemoryVectorOutboxDispatcher(repository=repository, queue=queue)  # type: ignore[arg-type]

    assert await dispatcher.dispatch_once() == 1
    assert repository.marked == ["event-1"]
    assert queue.messages == [MemoryVectorQueueMessage.from_event(_event())]


@pytest.mark.asyncio
async def test_vector_worker_embeds_current_postgres_document_then_acks() -> None:
    document = AgentMemoryVectorDocument(
        memory_id="mem-1",
        active=True,
        scope="workspace",
        user_id="user-1",
        workspace_id="workspace-1",
        kind="constraint",
        memory_key="reply_language",
        display_text="始终使用中文回答",
        topics=["产品设计"],
        version=2,
    )
    queue = _Queue(messages=[MemoryVectorQueueMessage.from_event(_event())], acknowledged=[])
    completed: list[str] = []

    class Repository:
        async def claim_vector_index(self, **_: object) -> MemoryVectorIndexClaim:
            return MemoryVectorIndexClaim(status="claimed", document=document)

        async def renew_vector_index_lease(self, **_: object) -> bool:
            return True

        async def complete_vector_index(
            self,
            *,
            event: MemoryVectorOutboxEvent,
            **_: object,
        ) -> None:
            completed.append(event.event_id)

    class Embedder:
        received: list[str] = []

        async def embed_text(self, text: str) -> SimpleNamespace:
            self.received.append(text)
            return SimpleNamespace(vector=[1.0, 0.0, 0.0])

    embedder = Embedder()
    vectors = _VectorStore()
    worker = MemoryVectorWorker(
        worker_id="worker-1",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        vector_store=vectors,  # type: ignore[arg-type]
        lease_seconds=60,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is True
    assert vectors.ensured == 1
    assert len(vectors.records) == 1
    record = vectors.records[0]
    assert record.memory_id == "mem-1"
    assert record.memory_version == 2
    assert "记忆键：reply_language" in embedder.received[0]
    assert completed == ["event-1"]
    assert queue.acknowledged == ["vector-1"]


@pytest.mark.asyncio
async def test_vector_worker_removes_an_inactive_memory_without_embedding() -> None:
    queue = _Queue(messages=[MemoryVectorQueueMessage.from_event(_event())], acknowledged=[])
    vectors = _VectorStore()

    class Repository:
        async def claim_vector_index(self, **_: object) -> MemoryVectorIndexClaim:
            return MemoryVectorIndexClaim(
                status="claimed",
                document=AgentMemoryVectorDocument(memory_id="mem-1", active=False),
            )

        async def renew_vector_index_lease(self, **_: object) -> bool:
            return True

        async def complete_vector_index(self, **_: object) -> None:
            return None

    class Embedder:
        async def embed_text(self, text: str) -> SimpleNamespace:
            raise AssertionError(f"inactive memory must not be embedded: {text}")

    worker = MemoryVectorWorker(
        worker_id="worker-1",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        embedder=Embedder(),  # type: ignore[arg-type]
        vector_store=vectors,  # type: ignore[arg-type]
        lease_seconds=60,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is True
    assert vectors.deleted == ["mem-1"]
    assert queue.acknowledged == ["vector-1"]


@pytest.mark.asyncio
async def test_vector_worker_leaves_lease_held_delivery_pending() -> None:
    queue = _Queue(messages=[MemoryVectorQueueMessage.from_event(_event())], acknowledged=[])

    class Repository:
        async def claim_vector_index(self, **_: object) -> MemoryVectorIndexClaim:
            return MemoryVectorIndexClaim(status="lease_held")

    worker = MemoryVectorWorker(
        worker_id="worker-2",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,  # type: ignore[arg-type]
        embedder=object(),  # type: ignore[arg-type]
        vector_store=object(),  # type: ignore[arg-type]
        lease_seconds=60,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is False
    assert queue.acknowledged == []
