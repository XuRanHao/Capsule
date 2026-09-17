import pytest

from capsule.agent.contracts import MemoryWrite
from capsule.db.agent_memory import PostgresAgentMemoryStore


@pytest.mark.asyncio
async def test_postgres_memory_store_reads_context_and_ignores_legacy_direct_writes() -> None:
    class Repository:
        calls: list[dict[str, object]] = []

        async def retrieve_memory_context(self, **kwargs: object) -> list[dict[str, object]]:
            self.calls.append(kwargs)
            return [{"scope": "workspace", "text": "项目使用中文"}]

    repository = Repository()
    store = PostgresAgentMemoryStore(repository, per_scope_limit=3)  # type: ignore[arg-type]

    loaded = await store.load(
        user_id="user-1",
        workspace_id="workspace-1",
        query="语言偏好",
    )
    await store.save(
        user_id="user-1",
        workspace_id="workspace-1",
        writes=[MemoryWrite(key="legacy", value="ignored")],
    )

    assert loaded == [{"scope": "workspace", "text": "项目使用中文"}]
    assert repository.calls == [
        {
            "user_id": "user-1",
            "workspace_id": "workspace-1",
            "query": "语言偏好",
            "per_scope_limit": 3,
        }
    ]
