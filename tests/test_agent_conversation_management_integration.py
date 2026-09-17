import os
from uuid import uuid4

import pytest
from sqlalchemy import delete

from capsule.config import Settings
from capsule.db.agent_memory import AgentConversationRepository, AgentThreadStateError
from capsule.db.models import AgentMemoryOutbox, Workspace
from capsule.db.session import Database


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_POSTGRES_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_POSTGRES_INTEGRATION=1 to exercise conversation lifecycle persistence",
)
async def test_conversation_management_preserves_history_and_stops_deleted_work() -> None:
    suffix = uuid4().hex
    workspace_id = f"conversation-{suffix}"
    user_id = f"conversation-user-{suffix}"
    database = Database(Settings())
    repository = AgentConversationRepository(database)
    try:
        async with database.session() as session, session.begin():
            session.add(Workspace(workspace_id=workspace_id, name="会话管理集成测试"))
        thread = await repository.create_thread(
            user_id=user_id,
            workspace_id=workspace_id,
            title="第 100% 次设计会话",
        )
        await repository.append_message(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            role="user",
            content="先保存这条会话。",
            request_id=f"message-{suffix}",
        )
        event = await repository.enqueue_consolidation_if_needed(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            token_threshold=1,
        )
        assert event is not None

        renamed = await repository.rename_thread(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
            title="  第 100% 次设计会话（已命名）  ",
        )
        assert renamed.title == "第 100% 次设计会话（已命名）"
        searched = await repository.list_threads(
            user_id=user_id,
            workspace_id=workspace_id,
            query="100%",
        )
        assert [item.thread_id for item in searched] == [thread.thread_id]

        archived = await repository.archive_thread(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        assert archived.status == "archived"
        with pytest.raises(AgentThreadStateError):
            await repository.append_message(
                thread_id=thread.thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
                role="user",
                content="归档后不能继续。",
                request_id=f"archived-{suffix}",
            )

        restored = await repository.restore_thread(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        assert restored.status == "active"
        deleted = await repository.soft_delete_thread(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        assert deleted.status == "deleted"
        assert deleted.deleted_at is not None
        assert await repository.list_threads(
            user_id=user_id,
            workspace_id=workspace_id,
        ) == []
        assert [item.thread_id for item in await repository.list_threads(
            user_id=user_id,
            workspace_id=workspace_id,
            include_deleted=True,
        )] == [thread.thread_id]
        with pytest.raises(ValueError, match="deleted"):
            await repository.list_messages(
                thread_id=thread.thread_id,
                user_id=user_id,
                workspace_id=workspace_id,
            )
        skipped = await repository.claim_consolidation(
            event=event,
            worker_id="lifecycle-worker",
            lease_seconds=30,
        )
        assert skipped.status == "already_consolidated"

        reactivated = await repository.restore_thread(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        assert reactivated.status == "active"
        assert reactivated.deleted_at is None
        messages = await repository.list_messages(
            thread_id=thread.thread_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        assert [item.content for item in messages] == ["先保存这条会话。"]
    finally:
        async with database.session() as session, session.begin():
            await session.execute(
                delete(AgentMemoryOutbox).where(
                    AgentMemoryOutbox.workspace_id == workspace_id
                )
            )
            workspace = await session.get(Workspace, workspace_id)
            if workspace is not None:
                await session.delete(workspace)
        await database.dispose()
