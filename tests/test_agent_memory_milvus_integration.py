import os
from uuid import uuid4

import pytest

from capsule.config import Settings
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore, AgentMemoryVectorRecord


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_MILVUS_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_MILVUS_INTEGRATION=1 to query a real collection",
)
async def test_real_memory_collection_supports_scoped_upsert_search_and_delete() -> None:
    collection = f"capsule_memory_test_{uuid4().hex}"
    settings = Settings(
        embedding_dimension=3,
        agent_memory_milvus_collection=collection,
    )
    store = AgentMemoryMilvusStore(settings)
    try:
        assert await store.ensure_collection() is True
        await store.aupsert(
            [
                AgentMemoryVectorRecord(
                    memory_id="mem-1",
                    scope="workspace",
                    user_id="user-1",
                    workspace_id="workspace-1",
                    kind="constraint",
                    memory_version=1,
                    vector=[1.0, 0.0, 0.0],
                ),
                AgentMemoryVectorRecord(
                    memory_id="mem-2",
                    scope="workspace",
                    user_id="user-2",
                    workspace_id="workspace-1",
                    kind="constraint",
                    memory_version=1,
                    vector=[1.0, 0.0, 0.0],
                ),
            ]
        )
        hits = await store.search(
            vector=[1.0, 0.0, 0.0],
            scope="workspace",
            user_id="user-1",
            workspace_id="workspace-1",
            limit=10,
        )

        assert [(item.memory_id, item.memory_version) for item in hits] == [("mem-1", 1)]
        assert await store.delete_workspace("workspace-1") >= 2
    finally:
        store._client.drop_collection(collection_name=collection)  # noqa: SLF001
