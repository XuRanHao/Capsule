import asyncio
from datetime import UTC, datetime

import pytest

from capsule.agent.memory_contracts import ConversationSummary, MemoryConsolidation
from capsule.agent.memory_worker import (
    InMemoryMemoryQueue,
    MemoryOutboxDispatcher,
    MemoryQueueMessage,
    MemoryWorker,
)
from capsule.db.agent_memory import (
    AgentThreadRecord,
    ClaimedMemoryConsolidation,
    MemoryConsolidationClaim,
    MemoryOutboxEvent,
)


def _event() -> MemoryOutboxEvent:
    return MemoryOutboxEvent(
        event_id="event-1",
        thread_id="thread-1",
        user_id="user-1",
        workspace_id="workspace-1",
        through_sequence=2,
    )


def _claimed() -> ClaimedMemoryConsolidation:
    return ClaimedMemoryConsolidation(
        event=_event(),
        messages=[],
        thread=AgentThreadRecord(
            thread_id="thread-1",
            user_id="user-1",
            workspace_id="workspace-1",
            title="新会话",
            status="active",
            summary=None,
            summary_topic=None,
            summary_covered_sequence=0,
            last_consolidated_sequence=0,
            memory_revision=0,
            last_message_at=None,
        ),
        active_topics=[],
    )


class _TrackingQueue(InMemoryMemoryQueue):
    def __init__(self) -> None:
        super().__init__()
        self.acknowledged: list[str] = []

    async def acknowledge(self, delivery):  # type: ignore[no-untyped-def]
        self.acknowledged.append(delivery.receipt)
        await super().acknowledge(delivery)


@pytest.mark.asyncio
async def test_memory_queue_round_trips_event_fields() -> None:
    queue = InMemoryMemoryQueue()
    await queue.start()
    await queue.publish(MemoryQueueMessage.from_event(_event()))

    delivery = await queue.receive()

    assert delivery.message.to_event() == _event()
    await queue.acknowledge(delivery)


@pytest.mark.asyncio
async def test_outbox_dispatcher_marks_event_only_after_queue_publish() -> None:
    class Repository:
        marked: list[str] = []

        async def pending_outbox_events(self, *, limit: int):
            assert limit == 100
            return [_event()]

        async def mark_outbox_published(self, *, event_id: str):
            self.marked.append(event_id)

    repository = Repository()
    queue = InMemoryMemoryQueue()
    dispatcher = MemoryOutboxDispatcher(repository=repository, queue=queue)  # type: ignore[arg-type]

    assert await dispatcher.dispatch_once() == 1
    assert repository.marked == ["event-1"]
    assert (await queue.receive()).message.event_id == "event-1"


@pytest.mark.asyncio
async def test_worker_acknowledges_delivery_after_consolidation_commits() -> None:
    queue = _TrackingQueue()
    await queue.publish(MemoryQueueMessage.from_event(_event()))
    calls: list[str] = []

    class Repository:
        async def claim_consolidation(self, **_: object):
            calls.append("claim")
            return MemoryConsolidationClaim(status="claimed", consolidation=_claimed())

        async def renew_consolidation_lease(self, **_: object) -> bool:
            return True

        async def complete_consolidation(self, **_: object):
            calls.append("complete")

    class Consolidator:
        async def consolidate(self, claimed: object) -> MemoryConsolidation:
            assert claimed is not None
            calls.append("consolidate")
            return MemoryConsolidation(
                summary=ConversationSummary(summary="摘要", covered_sequence=2)
            )

    worker = MemoryWorker(
        worker_id="worker-1",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,
        consolidator=Consolidator(),  # type: ignore[arg-type]
        lease_seconds=60,
        max_active_topics=5,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is True
    assert calls == ["claim", "consolidate", "complete"]
    assert queue.acknowledged == ["memory-1"]


@pytest.mark.asyncio
async def test_worker_keeps_delivery_pending_while_another_worker_holds_the_lease() -> None:
    queue = _TrackingQueue()
    await queue.publish(MemoryQueueMessage.from_event(_event()))

    class Repository:
        async def claim_consolidation(self, **_: object) -> MemoryConsolidationClaim:
            return MemoryConsolidationClaim(status="lease_held")

    class Consolidator:
        async def consolidate(self, claimed: object) -> MemoryConsolidation:
            raise AssertionError(f"must not consolidate while locked: {claimed}")

    worker = MemoryWorker(
        worker_id="worker-2",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,
        consolidator=Consolidator(),  # type: ignore[arg-type]
        lease_seconds=60,
        max_active_topics=5,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is False
    assert queue.acknowledged == []


@pytest.mark.asyncio
async def test_worker_acknowledges_an_already_completed_delivery() -> None:
    queue = _TrackingQueue()
    await queue.publish(MemoryQueueMessage.from_event(_event()))

    class Repository:
        async def claim_consolidation(self, **_: object) -> MemoryConsolidationClaim:
            return MemoryConsolidationClaim(status="already_consolidated")

    class Consolidator:
        async def consolidate(self, claimed: object) -> MemoryConsolidation:
            raise AssertionError(f"must not consolidate completed work: {claimed}")

    worker = MemoryWorker(
        worker_id="worker-2",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,
        consolidator=Consolidator(),  # type: ignore[arg-type]
        lease_seconds=60,
        max_active_topics=5,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is False
    assert queue.acknowledged == ["memory-1"]


@pytest.mark.asyncio
async def test_worker_renews_its_lease_during_slow_model_work() -> None:
    queue = _TrackingQueue()
    await queue.publish(MemoryQueueMessage.from_event(_event()))
    renewals: list[datetime] = []

    class Repository:
        async def claim_consolidation(self, **_: object) -> MemoryConsolidationClaim:
            return MemoryConsolidationClaim(status="claimed", consolidation=_claimed())

        async def renew_consolidation_lease(self, **_: object) -> bool:
            renewals.append(datetime.now(UTC))
            return True

        async def complete_consolidation(self, **_: object) -> None:
            return None

    class Consolidator:
        async def consolidate(self, claimed: object) -> MemoryConsolidation:
            assert claimed is not None
            await asyncio.sleep(0.15)
            return MemoryConsolidation(
                summary=ConversationSummary(summary="摘要", covered_sequence=2)
            )

    worker = MemoryWorker(
        worker_id="worker-3",
        repository=Repository(),  # type: ignore[arg-type]
        queue=queue,
        consolidator=Consolidator(),  # type: ignore[arg-type]
        lease_seconds=0.1,
        max_active_topics=5,
    )

    assert await asyncio.wait_for(worker.run_once(), timeout=1) is True
    assert renewals
