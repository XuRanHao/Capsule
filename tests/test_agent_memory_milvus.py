from typing import Any

import pytest

from capsule.config import Settings
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore, AgentMemoryVectorRecord


class _Client:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.deleted: list[dict[str, Any]] = []
        self.flushes: list[str] = []

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.kwargs = kwargs
        return [
            [
                {
                    "id": "mem-1",
                    "distance": 0.91,
                    "entity": {"memory_version": 4},
                }
            ]
        ]

    def upsert(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def flush(self, *, collection_name: str) -> None:
        self.flushes.append(collection_name)

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.deleted.append(kwargs)
        return {"delete_count": 2}


@pytest.mark.asyncio
async def test_memory_milvus_search_scopes_user_and_workspace_and_keeps_version() -> None:
    client = _Client()
    store = AgentMemoryMilvusStore(Settings(embedding_dimension=3), client=client)

    hits = await store.search(
        vector=[1.0, 0.0, 0.0],
        scope="workspace",
        user_id='user_"one',
        workspace_id='workspace_"one',
        limit=12,
    )

    assert client.kwargs["collection_name"] == "agent_memory_embeddings_seed16_1024_v1"
    assert client.kwargs["filter"] == (
        'scope == "workspace" and user_id == "user_\\"one" '
        'and workspace_id == "workspace_\\"one"'
    )
    assert client.kwargs["output_fields"] == ["memory_id", "memory_version"]
    assert hits[0].memory_id == "mem-1"
    assert hits[0].memory_version == 4


@pytest.mark.asyncio
async def test_memory_milvus_upserts_current_version_and_deletes_a_workspace() -> None:
    client = _Client()
    store = AgentMemoryMilvusStore(Settings(embedding_dimension=3), client=client)

    await store.aupsert(
        [
            AgentMemoryVectorRecord(
                memory_id="mem-1",
                scope="workspace",
                user_id="user-1",
                workspace_id="workspace-1",
                kind="constraint",
                memory_version=2,
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    deleted = await store.delete_workspace("workspace-1")

    assert client.kwargs["data"][0]["memory_version"] == 2
    assert client.flushes == ["agent_memory_embeddings_seed16_1024_v1"]
    assert deleted == 2
    assert client.deleted[0]["filter"] == (
        'scope == "workspace" and workspace_id == "workspace-1"'
    )
