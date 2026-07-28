import os

import pytest

from capsule.config import Settings
from capsule.vectorstore.milvus import MilvusVectorStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_MILVUS_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_MILVUS_INTEGRATION=1 to query a real collection",
)
async def test_real_milvus_collection_accepts_scoped_search() -> None:
    settings = Settings()
    store = MilvusVectorStore(settings)

    hits = await store.search(
        vector=[1.0] + [0.0] * (settings.embedding_dimension - 1),
        workspace_id=os.getenv("CAPSULE_TEST_WORKSPACE_ID", "workspace_demo"),
        embedding_type="native_multimodal",
        asset_types=(),
        limit=1,
    )

    assert isinstance(hits, list)
