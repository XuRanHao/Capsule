from dataclasses import replace
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from capsule.agent.runtime import AgentRuntime
from capsule.api.app import create_app
from capsule.config import Settings
from capsule.db.agent_memory import AgentConversationRepository, AgentThreadRecord


class _ConversationRepositoryStub(AgentConversationRepository):
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.thread = AgentThreadRecord(
            thread_id="thread-api",
            user_id="user-api",
            workspace_id="workspace-api",
            title="初始标题",
            status="active",
            summary=None,
            summary_topic=None,
            summary_covered_sequence=0,
            last_consolidated_sequence=0,
            memory_revision=0,
            last_message_at=now,
        )
        self.list_arguments: dict[str, object] | None = None

    async def list_threads(self, **kwargs: object) -> list[AgentThreadRecord]:
        self.list_arguments = kwargs
        return [self.thread]

    async def rename_thread(self, *, title: str, **_: object) -> AgentThreadRecord:
        self.thread = replace(self.thread, title=title.strip())
        return self.thread

    async def archive_thread(self, **_: object) -> AgentThreadRecord:
        self.thread = replace(self.thread, status="archived")
        return self.thread

    async def restore_thread(self, **_: object) -> AgentThreadRecord:
        self.thread = replace(self.thread, status="active", deleted_at=None)
        return self.thread

    async def soft_delete_thread(self, **_: object) -> AgentThreadRecord:
        self.thread = replace(
            self.thread,
            status="deleted",
            deleted_at=datetime.now(UTC),
        )
        return self.thread


@pytest.mark.asyncio
async def test_agent_conversation_management_routes_expose_lifecycle_controls() -> None:
    runtime = AgentRuntime()
    repository = _ConversationRepositoryStub()
    app = create_app(settings=Settings(), agent_runtime=runtime)
    async with app.router.lifespan_context(app):
        app.state.agent_conversation_repository = repository
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            listed = await client.get(
                "/api/v1/agent/threads",
                params={
                    "user_id": "user-api",
                    "workspace_id": "workspace-api",
                    "status": "archived",
                    "query": "标题",
                    "include_deleted": "true",
                },
            )
            renamed = await client.patch(
                "/api/v1/agent/threads/thread-api",
                json={
                    "user_id": "user-api",
                    "workspace_id": "workspace-api",
                    "title": "新标题",
                },
            )
            archived = await client.post(
                "/api/v1/agent/threads/thread-api/archive",
                params={"user_id": "user-api", "workspace_id": "workspace-api"},
            )
            restored = await client.post(
                "/api/v1/agent/threads/thread-api/restore",
                params={"user_id": "user-api", "workspace_id": "workspace-api"},
            )
            deleted = await client.delete(
                "/api/v1/agent/threads/thread-api",
                params={"user_id": "user-api", "workspace_id": "workspace-api"},
            )

    assert listed.status_code == 200
    assert repository.list_arguments == {
        "user_id": "user-api",
        "workspace_id": "workspace-api",
        "limit": 50,
        "status": "archived",
        "query": "标题",
        "include_deleted": True,
    }
    assert renamed.json()["title"] == "新标题"
    assert archived.json()["status"] == "archived"
    assert restored.json()["status"] == "active"
    assert deleted.json()["status"] == "deleted"
    assert deleted.json()["deleted_at"] is not None
