from typing import Any

from capsule.config import Settings
from capsule.vectorstore.milvus import MilvusVectorStore


class FakeMilvusClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]:
        self.kwargs = kwargs
        return [
            [
                {
                    "id": "emb_1",
                    "distance": 0.91,
                    "entity": {
                        "asset_id": "asset_1",
                        "source_file_id": "source_1",
                        "asset_type": "image",
                        "embedding_revision": 2,
                    },
                }
            ]
        ]


async def test_milvus_search_builds_scoped_filter_and_parses_hits() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(
        Settings(embedding_dimension=3),
        client=client,
    )

    hits = await store.search(
        vector=[1.0, 0.0, 0.0],
        workspace_id='workspace_"demo',
        embedding_type="native_multimodal",
        asset_types=("image", "video_segment"),
        limit=60,
    )

    expression = client.kwargs["filter"]
    assert 'workspace_id == "workspace_\\"demo"' in expression
    assert 'embedding_type == "native_multimodal"' in expression
    assert 'asset_type in ["image", "video_segment"]' in expression
    assert client.kwargs["limit"] == 60
    assert hits[0].asset_id == "asset_1"
    assert hits[0].embedding_revision == 2
    assert hits[0].similarity == 0.91
