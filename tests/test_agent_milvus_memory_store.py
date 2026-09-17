from types import SimpleNamespace

import pytest

from capsule.agent.milvus_memory_store import MilvusAgentMemoryStore
from capsule.vectorstore.agent_memory import AgentMemoryVectorHit


@pytest.mark.asyncio
async def test_memory_store_uses_ann_candidates_then_postgres_hydration() -> None:
    class Repository:
        received: dict[str, object] | None = None

        async def retrieve_memory_context_from_vectors(self, **kwargs: object):
            self.received = kwargs
            return [{"memory_id": "mem-workspace", "text": "使用中文"}]

    class Embedder:
        async def embed_text(self, text: str) -> SimpleNamespace:
            assert text == "怎么回复用户"
            return SimpleNamespace(vector=[1.0, 0.0, 0.0])

    class Vectors:
        async def search(self, *, scope: str, **_: object):
            if scope == "workspace":
                return [AgentMemoryVectorHit("mem-workspace", 0.91, 2)]
            return [AgentMemoryVectorHit("mem-global", 0.83, 5)]

    repository = Repository()
    store = MilvusAgentMemoryStore(
        repository=repository,  # type: ignore[arg-type]
        embedder=Embedder(),  # type: ignore[arg-type]
        vector_store=Vectors(),  # type: ignore[arg-type]
        per_scope_limit=3,
        candidate_multiplier=4,
    )

    result = await store.load(
        user_id="user-1",
        workspace_id="workspace-1",
        query="怎么回复用户",
    )

    assert result == [{"memory_id": "mem-workspace", "text": "使用中文"}]
    assert repository.received == {
        "user_id": "user-1",
        "workspace_id": "workspace-1",
        "workspace_candidates": [("mem-workspace", 0.91, 2)],
        "global_candidates": [("mem-global", 0.83, 5)],
        "per_scope_limit": 3,
    }


@pytest.mark.asyncio
async def test_memory_store_falls_back_to_lexical_reader_when_vector_recall_fails() -> None:
    class Repository:
        async def retrieve_memory_context_from_vectors(self, **_: object):
            raise AssertionError("vector failure must not hydrate candidates")

        async def retrieve_memory_context(self, **kwargs: object):
            assert kwargs["query"] == "项目约束"
            return [{"memory_id": "mem-lexical", "text": "项目约束"}]

    class Embedder:
        async def embed_text(self, text: str) -> SimpleNamespace:
            return SimpleNamespace(vector=[1.0, 0.0, 0.0])

    class Vectors:
        async def search(self, **_: object):
            raise RuntimeError("Milvus unavailable")

    store = MilvusAgentMemoryStore(
        repository=Repository(),  # type: ignore[arg-type]
        embedder=Embedder(),  # type: ignore[arg-type]
        vector_store=Vectors(),  # type: ignore[arg-type]
        per_scope_limit=3,
        candidate_multiplier=4,
    )

    assert await store.load(
        user_id="user-1",
        workspace_id="workspace-1",
        query="项目约束",
    ) == [{"memory_id": "mem-lexical", "text": "项目约束"}]
