from typing import Any

from capsule.config import Settings
from capsule.search.models import SearchFilters
from capsule.vectorstore.milvus import MilvusVectorStore, VectorRecord


class FakeMilvusClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.delete_calls: list[dict[str, Any]] = []
        self.flush_calls: list[str] = []

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

    def upsert(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def query(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.kwargs = kwargs
        row: dict[str, Any] = {"embedding_id": "emb_1"}
        if "vector" in kwargs.get("output_fields", []):
            row["vector"] = [0.0, 1.0, 2.0]
        return [row]

    def delete(self, **kwargs: Any) -> dict[str, int]:
        self.kwargs = kwargs
        self.delete_calls.append(kwargs)
        return {"delete_count": 2}

    def flush(self, *, collection_name: str) -> None:
        self.flush_calls.append(collection_name)


class FakeBootstrapMilvusClient(FakeMilvusClient):
    def __init__(
        self,
        *,
        exists: bool = False,
        dimension: int = 3,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.collections = (
            {
                "asset_embeddings_seed16_1024",
                "asset_embeddings_seed16_1024_fused_n03_v1",
            }
            if exists
            else set()
        )
        self.created: list[dict[str, Any]] = []
        self.loaded: list[str] = []

    def has_collection(self, *, collection_name: str) -> bool:
        self.kwargs["has_collection"] = collection_name
        return collection_name in self.collections

    def describe_collection(self, *, collection_name: str) -> dict[str, Any]:
        self.kwargs["describe_collection"] = collection_name
        return {
            "fields": [
                {"name": field_name}
                for field_name in (
                    "embedding_id",
                    "workspace_id",
                    "project_id",
                    "asset_id",
                    "source_file_id",
                    "asset_type",
                    "file_type",
                    "embedding_type",
                    "model_name",
                    "embedding_revision",
                    "created_at_ts",
                )
            ]
            + [
                {
                    "name": "vector",
                    "params": {"dim": self.dimension},
                }
            ]
        }

    def create_collection(self, **kwargs: Any) -> None:
        self.created.append(kwargs)
        self.collections.add(kwargs["collection_name"])

    def load_collection(self, *, collection_name: str) -> None:
        self.loaded.append(collection_name)


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
        filters=SearchFilters(
            project_id="project_demo",
            asset_type=["image", "video_segment"],
            file_type=[".png"],
            source_file_id=["source_1"],
            model_name=["seed-1.6"],
        ),
        limit=60,
    )

    expression = client.kwargs["filter"]
    assert 'workspace_id == "workspace_\\"demo"' in expression
    assert 'embedding_type == "native_multimodal"' in expression
    assert 'project_id == "project_demo"' in expression
    assert 'asset_type in ["image", "video_segment"]' in expression
    assert 'file_type in [".png"]' in expression
    assert 'source_file_id in ["source_1"]' in expression
    assert 'model_name in ["seed-1.6"]' in expression
    assert client.kwargs["search_params"]["params"]["ef"] == 128
    assert client.kwargs["limit"] == 60
    assert client.kwargs["collection_name"] == "asset_embeddings_seed16_1024"
    assert hits[0].asset_id == "asset_1"
    assert hits[0].embedding_revision == 2
    assert hits[0].similarity == 0.91


async def test_milvus_search_routes_dimension_channels_to_fused_collection() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    hits = await store.search(
        vector=[1.0, 0.0, 0.0],
        workspace_id="workspace_demo",
        embedding_type="visual_style",
        filters=SearchFilters(),
        limit=10,
    )

    assert client.kwargs["collection_name"] == (
        "asset_embeddings_seed16_1024_fused_n03_v1"
    )
    assert client.kwargs["filter"] == (
        'workspace_id == "workspace_demo" and embedding_type == "visual_style"'
    )
    assert hits[0].embedding_id == "emb_1"


def test_search_filter_accepts_legacy_model_field_but_serializes_database_name() -> None:
    filters = SearchFilters.model_validate(
        {"embedding_model_version": ["doubao-embedding-vision-250615"]}
    )

    assert filters.model_name == ["doubao-embedding-vision-250615"]
    assert filters.model_dump() == {
        "project_id": None,
        "asset_type": [],
        "file_type": [],
        "source_file_id": [],
        "created_at_from": None,
        "created_at_to": None,
        "model_name": ["doubao-embedding-vision-250615"],
        "favorite": None,
    }


async def test_milvus_async_upsert_uses_embedding_id_as_primary_key() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    await store.aupsert(
        [
            VectorRecord(
                embedding_id="emb_1",
                workspace_id="workspace_demo",
                project_id="project_default",
                asset_id="asset_1",
                source_file_id="src_1",
                asset_type="markdown_block",
                file_type=".md",
                embedding_type="native_multimodal",
                model_name="doubao-embedding-vision-250615",
                embedding_revision=2,
                created_at_ts=1_700_000_000,
                vector=[0.0, 1.0, 2.0],
            )
        ]
    )

    assert client.kwargs["collection_name"] == "asset_embeddings_seed16_1024"
    assert client.kwargs["data"] == [
        {
            "embedding_id": "emb_1",
            "workspace_id": "workspace_demo",
            "project_id": "project_default",
            "asset_id": "asset_1",
            "source_file_id": "src_1",
            "asset_type": "markdown_block",
            "file_type": ".md",
            "embedding_type": "native_multimodal",
            "model_name": "doubao-embedding-vision-250615",
            "embedding_revision": 2,
            "created_at_ts": 1_700_000_000,
            "vector": [0.0, 1.0, 2.0],
        }
    ]


async def test_milvus_fused_upsert_keeps_dimension_embedding_id() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    await store.aupsert_fused(
        [
            VectorRecord(
                embedding_id="emb_dimension_1",
                workspace_id="workspace_demo",
                project_id="project_default",
                asset_id="asset_1",
                source_file_id="src_1",
                asset_type="image",
                file_type=".png",
                embedding_type="visual_style",
                model_name="doubao-embedding-vision-250615",
                embedding_revision=2,
                created_at_ts=1_700_000_000,
                vector=[0.3, 0.7, 0.0],
            )
        ]
    )

    assert client.kwargs["collection_name"] == (
        "asset_embeddings_seed16_1024_fused_n03_v1"
    )
    assert client.kwargs["data"][0]["embedding_id"] == "emb_dimension_1"
    assert client.kwargs["data"][0]["embedding_type"] == "visual_style"
    assert client.flush_calls == ["asset_embeddings_seed16_1024_fused_n03_v1"]


async def test_milvus_fetch_vectors_queries_exact_embedding_ids() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    vectors = await store.fetch_vectors(["emb_1", "emb_missing"])

    assert vectors == {"emb_1": [0.0, 1.0, 2.0]}
    assert client.kwargs["collection_name"] == "asset_embeddings_seed16_1024"
    assert client.kwargs["filter"] == 'embedding_id in ["emb_1", "emb_missing"]'
    assert client.kwargs["output_fields"] == ["embedding_id", "vector"]
    assert client.kwargs["consistency_level"] == "Strong"


async def test_milvus_fetch_fused_vector_ids_checks_derived_collection() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    visible = await store.fetch_fused_vector_ids(["emb_1", "emb_missing"])

    assert visible == {"emb_1"}
    assert client.kwargs["collection_name"] == (
        "asset_embeddings_seed16_1024_fused_n03_v1"
    )
    assert client.kwargs["filter"] == 'embedding_id in ["emb_1", "emb_missing"]'
    assert client.kwargs["output_fields"] == ["embedding_id"]
    assert client.kwargs["consistency_level"] == "Strong"


async def test_milvus_workspace_delete_clears_raw_and_fused_collections() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    deleted = await store.delete_workspace('workspace_"demo')

    assert deleted == 4
    assert client.delete_calls == [
        {
            "collection_name": "asset_embeddings_seed16_1024",
            "filter": 'workspace_id == "workspace_\\"demo"',
        },
        {
            "collection_name": "asset_embeddings_seed16_1024_fused_n03_v1",
            "filter": 'workspace_id == "workspace_\\"demo"',
        },
    ]


async def test_milvus_delete_all_clears_raw_and_fused_collections() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    deleted = await store.delete_all()

    assert deleted == 4
    assert [call["collection_name"] for call in client.delete_calls] == [
        "asset_embeddings_seed16_1024",
        "asset_embeddings_seed16_1024_fused_n03_v1",
    ]
    assert all(call["filter"] == 'embedding_id != ""' for call in client.delete_calls)


async def test_milvus_bootstrap_creates_and_loads_frozen_collection() -> None:
    client = FakeBootstrapMilvusClient()
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    created = await store.ensure_collection()

    assert created is True
    assert [item["collection_name"] for item in client.created] == [
        "asset_embeddings_seed16_1024",
        "asset_embeddings_seed16_1024_fused_n03_v1",
    ]
    assert client.loaded == [
        "asset_embeddings_seed16_1024",
        "asset_embeddings_seed16_1024_fused_n03_v1",
    ]


async def test_milvus_bootstrap_reuses_matching_collection() -> None:
    client = FakeBootstrapMilvusClient(exists=True, dimension=3)
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    created = await store.ensure_collection()

    assert created is False
    assert client.created == []
    assert client.loaded == [
        "asset_embeddings_seed16_1024",
        "asset_embeddings_seed16_1024_fused_n03_v1",
    ]


async def test_milvus_bootstrap_adds_fused_collection_beside_existing_raw() -> None:
    client = FakeBootstrapMilvusClient(exists=True, dimension=3)
    client.collections.remove("asset_embeddings_seed16_1024_fused_n03_v1")
    store = MilvusVectorStore(Settings(embedding_dimension=3), client=client)

    created = await store.ensure_collection()

    assert created is True
    assert [item["collection_name"] for item in client.created] == [
        "asset_embeddings_seed16_1024_fused_n03_v1"
    ]
    assert client.loaded == [
        "asset_embeddings_seed16_1024",
        "asset_embeddings_seed16_1024_fused_n03_v1",
    ]


async def test_milvus_bootstrap_rejects_dimension_mismatch() -> None:
    client = FakeBootstrapMilvusClient(exists=True, dimension=768)
    store = MilvusVectorStore(Settings(embedding_dimension=1024), client=client)

    try:
        await store.ensure_collection()
    except ValueError as exc:
        assert "has dimension 768, expected 1024" in str(exc)
    else:
        raise AssertionError("dimension mismatch should fail bootstrap")
