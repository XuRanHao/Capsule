from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import (
    Asset,
    ClusterCapsule,
    ClusterMembership,
    ClusterRun,
    EmbeddingRecord,
    SourceFile,
    Workspace,
)
from capsule.db.session import Database
from capsule.enums import ClusterRunStatus, EmbeddingStatus
from capsule.search.models import SearchFilters
from capsule.search.repositories import PostgresAssetSearchRepository


@pytest.mark.integration
async def test_search_hydration_uses_latest_asset_and_embedding_fields() -> None:
    database = Database(get_settings())
    suffix = uuid4().hex[:12]
    workspace_id = f"workspace_search_fields_{suffix}"
    source_file_id = f"src_search_fields_{suffix}"
    asset_id = f"asset_search_fields_{suffix}"
    embedding_id = f"emb_search_fields_{suffix}"
    cluster_run_id = f"run_search_fields_{suffix}"
    cluster_capsule_id = f"cc_search_fields_{suffix}"
    model_name = "doubao-embedding-vision-250615"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        async with database.session() as session, session.begin():
            session.add(Workspace(workspace_id=workspace_id, name=workspace_id))
            await session.flush()
            session.add(
                SourceFile(
                    source_file_id=source_file_id,
                    workspace_id=workspace_id,
                    project_id="project_default",
                    original_file_name="board.md",
                    file_type=".md",
                    mime_type="text/markdown",
                    relative_path="references/board.md",
                    file_tree_context=["references"],
                    storage_uri="file:///references/board.md",
                    sha256="a" * 64,
                    file_size_bytes=2048,
                    processing_status="completed",
                )
            )
            await session.flush()
            session.add(
                Asset(
                    asset_id=asset_id,
                    workspace_id=workspace_id,
                    project_id="project_default",
                    source_file_id=source_file_id,
                    asset_type="image",
                    file_name="sunset.png",
                    file_type=".png",
                    asset_key="image:0",
                    content_hash="b" * 64,
                    asset_name="黄昏",
                    asset_name_source="model",
                    asset_description="Markdown 中的一张黄昏图片",
                    asset_features={"mood_atmosphere": {"value": "宁静"}},
                    file_tree_context=["references"],
                    source_contexts=[
                        {
                            "text": "午后-黄昏",
                            "relation_type": "preceding_text",
                            "text_block_index": 2,
                        }
                    ],
                    file_info={"width": 1200, "height": 800},
                    source_locator={"block_index": 3},
                    preview_uri="s3://capsule/previews/sunset.jpg",
                    processing_status="completed",
                    feature_revision=2,
                    embedding_revision=3,
                )
            )
            await session.flush()
            session.add(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    workspace_id=workspace_id,
                    project_id="project_default",
                    asset_id=asset_id,
                    embedding_type="native_multimodal",
                    model_name=model_name,
                    dimension=1024,
                    source_content_hash="c" * 64,
                    embedding_source_mode="original_image",
                    milvus_collection="asset_embeddings_seed16_1024",
                    milvus_primary_key=embedding_id,
                    status=EmbeddingStatus.INDEXED.value,
                )
            )
            session.add(
                ClusterRun(
                    cluster_run_id=cluster_run_id,
                    workspace_id=workspace_id,
                    embedding_type="native_multimodal",
                    input_embedding_ids=[embedding_id],
                    dataset_hash="d" * 64,
                    sample_count=1,
                    preprocessing={},
                    parameters={},
                    cluster_count=1,
                    noise_count=0,
                    noise_ratio=0,
                    status=ClusterRunStatus.COMPLETED.value,
                )
            )
            await session.flush()
            session.add(
                ClusterCapsule(
                    cluster_capsule_id=cluster_capsule_id,
                    cluster_run_id=cluster_run_id,
                    workspace_id=workspace_id,
                    embedding_type="native_multimodal",
                    cluster_label=0,
                    model_generated_name="黄昏场景",
                    effective_name="黄昏场景",
                    model_generated_description="包含蓝紫色黄昏素材的聚类。",
                    effective_description="包含蓝紫色黄昏素材的聚类。",
                    keywords=["黄昏"],
                    common_features=["蓝紫色"],
                    internal_variance="low",
                    member_count=1,
                    average_membership_probability=0.95,
                    medoid_asset_id=asset_id,
                    representative_asset_ids=[asset_id],
                )
            )
            await session.flush()
            session.add(
                ClusterMembership(
                    cluster_run_id=cluster_run_id,
                    cluster_capsule_id=cluster_capsule_id,
                    asset_id=asset_id,
                    hdbscan_label=0,
                    membership_probability=0.95,
                    is_noise=False,
                    distance_to_representative=0,
                )
            )

        repository = PostgresAssetSearchRepository(database)
        records = await repository.get_by_ids(
            workspace_id=workspace_id,
            asset_ids=[asset_id],
            embedding_ids=[embedding_id],
            filters=SearchFilters(file_type=[".png"], model_name=[model_name]),
        )

        record = records[asset_id]
        assert record.file_type == ".png"
        assert record.source_file_type == ".md"
        assert record.file_tree_context == ["references"]
        assert record.source_contexts[0]["text"] == "午后-黄昏"
        assert record.embedding_revision == 3
        assert record.indexed_embedding_ids == frozenset({embedding_id})

        source_file_type_does_not_override_asset = await repository.get_by_ids(
            workspace_id=workspace_id,
            asset_ids=[asset_id],
            embedding_ids=[embedding_id],
            filters=SearchFilters(file_type=[".md"]),
        )
        assert source_file_type_does_not_override_asset == {}

        clusters = await repository.search_by_assets(
            workspace_id=workspace_id,
            asset_scores={asset_id: 0.8},
            embedding_types=["native_multimodal"],
            limit=5,
        )
        assert len(clusters) == 1
        assert clusters[0].cluster_capsule_id == cluster_capsule_id
        assert clusters[0].matched_asset_ids == [asset_id]
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()
