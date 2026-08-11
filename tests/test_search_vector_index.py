import math
from dataclasses import replace

import pytest

from capsule.enums import EmbeddingType
from capsule.pipeline.search_vector_index import (
    SearchVectorIndexMaterializer,
    build_fused_search_vector_record,
)
from capsule.vectorstore.milvus import VectorRecord


class FakeSearchVectorStore:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.upsert_calls: list[list[VectorRecord]] = []

    async def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return self.ensure_calls == 1

    async def aupsert_fused(self, records: list[VectorRecord]) -> None:
        self.upsert_calls.append(list(records))


def _record(
    *,
    embedding_id: str,
    asset_id: str,
    embedding_type: EmbeddingType,
    vector: list[float],
) -> VectorRecord:
    return VectorRecord(
        embedding_id=embedding_id,
        workspace_id="workspace_demo",
        project_id="project_default",
        asset_id=asset_id,
        source_file_id=f"source_{asset_id}",
        asset_type="image",
        file_type=".png",
        embedding_type=embedding_type.value,
        model_name="seed-test",
        embedding_revision=2,
        created_at_ts=1_700_000_000,
        vector=vector,
    )


def test_fused_search_record_reuses_dimension_identity_and_metadata() -> None:
    native = _record(
        embedding_id="emb_native",
        asset_id="asset_1",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        vector=[3.0, 0.0],
    )
    dimension = _record(
        embedding_id="emb_visual",
        asset_id="asset_1",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        vector=[0.0, 4.0],
    )

    fused = build_fused_search_vector_record(
        native_record=native,
        dimension_record=dimension,
    )

    assert replace(fused, vector=dimension.vector) == dimension
    assert fused.embedding_id == "emb_visual"
    assert fused.embedding_type == EmbeddingType.VISUAL_STYLE.value
    assert fused.vector == pytest.approx([0.3 / math.sqrt(0.58), 0.7 / math.sqrt(0.58)])
    assert math.isclose(
        sum(value * value for value in fused.vector),
        1.0,
        rel_tol=1e-6,
    )


@pytest.mark.asyncio
async def test_materializer_batches_current_pairs_into_destination_store() -> None:
    store = FakeSearchVectorStore()
    materializer = SearchVectorIndexMaterializer(store)
    native_records = [
        _record(
            embedding_id=f"native_{index}",
            asset_id=f"asset_{index}",
            embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
            vector=[1.0, float(index)],
        )
        for index in (1, 2)
    ]
    dimension_records = [
        _record(
            embedding_id=f"visual_{index}",
            asset_id=f"asset_{index}",
            embedding_type=EmbeddingType.VISUAL_STYLE,
            vector=[float(index), 1.0],
        )
        for index in (1, 2)
    ]

    result = await materializer.materialize(
        native_records=native_records,
        dimension_records=dimension_records,
    )

    assert result.requested_count == 2
    assert result.materialized_count == 2
    assert result.failed_count == 0
    assert result.embedding_ids == ("visual_1", "visual_2")
    assert store.ensure_calls == 1
    assert len(store.upsert_calls) == 1
    assert [record.embedding_id for record in store.upsert_calls[0]] == [
        "visual_1",
        "visual_2",
    ]


@pytest.mark.asyncio
async def test_materializer_reports_missing_or_stale_native_pair() -> None:
    store = FakeSearchVectorStore()
    materializer = SearchVectorIndexMaterializer(store)
    current_native = _record(
        embedding_id="native_1",
        asset_id="asset_1",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        vector=[1.0, 0.0],
    )
    current_dimension = _record(
        embedding_id="visual_1",
        asset_id="asset_1",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        vector=[0.0, 1.0],
    )
    missing_native_dimension = _record(
        embedding_id="visual_2",
        asset_id="asset_2",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        vector=[0.0, 1.0],
    )
    stale_native_dimension = replace(
        current_dimension,
        embedding_id="visual_stale",
        embedding_revision=3,
    )

    result = await materializer.materialize(
        native_records=[current_native],
        dimension_records=[
            current_dimension,
            missing_native_dimension,
            stale_native_dimension,
        ],
    )

    assert result.materialized_count == 1
    assert result.failed_count == 2
    assert [failure.embedding_id for failure in result.failures] == [
        "visual_2",
        "visual_stale",
    ]
    assert "missing" in result.failures[0].reason
    assert "embedding_revision" in result.failures[1].reason
    assert [record.embedding_id for record in store.upsert_calls[0]] == ["visual_1"]


@pytest.mark.asyncio
async def test_materializer_does_not_initialize_destination_for_empty_batch() -> None:
    store = FakeSearchVectorStore()
    result = await SearchVectorIndexMaterializer(store).materialize(
        native_records=[],
        dimension_records=[],
    )

    assert result.requested_count == 0
    assert result.materialized_count == 0
    assert store.ensure_calls == 0
    assert store.upsert_calls == []


def test_fused_search_record_rejects_cross_asset_pair() -> None:
    native = _record(
        embedding_id="native_1",
        asset_id="asset_1",
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        vector=[1.0, 0.0],
    )
    dimension = _record(
        embedding_id="visual_2",
        asset_id="asset_2",
        embedding_type=EmbeddingType.VISUAL_STYLE,
        vector=[0.0, 1.0],
    )

    with pytest.raises(ValueError, match="asset_id"):
        build_fused_search_vector_record(
            native_record=native,
            dimension_record=dimension,
        )
