import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from capsule.agent.memory_contracts import (
    ConversationSummary,
    MemoryCandidate,
    MemoryConsolidation,
    MemoryMutation,
)
from capsule.config import Settings
from capsule.db.agent_memory import AgentConversationRepository
from capsule.db.models import AgentMemory, AgentMemoryVectorOutbox, Workspace
from capsule.db.session import Database


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_POSTGRES_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_POSTGRES_INTEGRATION=1 to exercise memory vector persistence",
)
async def test_memory_mutation_creates_and_completes_a_durable_vector_outbox_event() -> None:
    suffix = uuid4().hex
    workspace_id = f"memory-vector-workspace-{suffix}"
    user_id = f"memory-vector-user-{suffix}"
    database = Database(Settings())
    repository = AgentConversationRepository(database)
    memory_id: str | None = None
    try:
        async with database.session() as session, session.begin():
            session.add(Workspace(workspace_id=workspace_id, name="记忆向量集成测试"))
        thread = await repository.create_thread(
            user_id=user_id,
            workspace_id=workspace_id,
            title="测试会话",
        )
        await repository.append_message(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            role="user",
            content="请始终使用中文回答。",
            request_id=f"request-{suffix}",
            turn_id=f"turn-{suffix}",
        )
        event = await repository.enqueue_consolidation_if_needed(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            token_threshold=1,
        )
        assert event is not None
        claim = await repository.claim_consolidation(
            event=event,
            worker_id="integration-worker",
            lease_seconds=30,
        )
        assert claim.consolidation is not None
        prepared = await repository.persist_short_term_consolidation(
            claimed=claim.consolidation,
            worker_id="integration-worker",
            summary=ConversationSummary(
                summary="用户要求中文回答。",
                topic="表达偏好",
                covered_sequence=event.through_sequence,
            ),
            max_active_topics=5,
        )
        await repository.complete_consolidation(
            claimed=prepared,
            worker_id="integration-worker",
            consolidation=MemoryConsolidation(
                summary=ConversationSummary(
                    summary="用户要求中文回答。",
                    topic="表达偏好",
                    covered_sequence=event.through_sequence,
                ),
                mutations=[
                    MemoryMutation(
                        action="create",
                        candidate=MemoryCandidate(
                            scope="workspace",
                            kind="reply_preference",
                            memory_key="language",
                            value={"language": "zh-CN"},
                            display_text="此工作区内始终使用中文回答。",
                            topics=["表达偏好"],
                            initial_confidence=0.9,
                            decay_rate=0.002,
                        ),
                    )
                ],
            ),
        )

        async with database.session() as session:
            memory_id = await session.scalar(
                select(AgentMemory.memory_id).where(
                    AgentMemory.workspace_id == workspace_id
                )
            )
        assert memory_id is not None
        vector_events = [
            item
            for item in await repository.pending_vector_outbox_events()
            if item.memory_id == memory_id
        ]
        assert len(vector_events) == 1
        vector_event = vector_events[0]
        claim = await repository.claim_vector_index(
            event=vector_event,
            worker_id="integration-vector-worker",
            lease_seconds=30,
        )
        assert claim.status == "claimed"
        assert claim.document is not None
        assert claim.document.active is True
        assert claim.document.memory_key == "language"
        await repository.complete_vector_index(
            event=vector_event,
            worker_id="integration-vector-worker",
        )
        assert all(
            item.memory_id != memory_id
            for item in await repository.pending_vector_outbox_events()
        )
    finally:
        async with database.session() as session, session.begin():
            if memory_id is not None:
                await session.execute(
                    delete(AgentMemoryVectorOutbox).where(
                        AgentMemoryVectorOutbox.memory_id == memory_id
                    )
                )
            workspace = await session.get(Workspace, workspace_id)
            if workspace is not None:
                await session.delete(workspace)
        await database.dispose()
