from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, text

from capsule.config import get_settings
from capsule.db.models import (
    Asset,
    ClusterRun,
    CurrentCluster,
    CurrentClusterMember,
    EmbeddingRecord,
    SourceFile,
    Workspace,
)
from capsule.db.repositories import CurrentClusterRepository
from capsule.db.session import Database
from capsule.enums import NewAssetClusterStatus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_new_asset_cluster_status_uses_latest_vectors_and_full_run_baseline() -> None:
    database = Database(get_settings())
    workspace_id = "workspace_new_asset_cluster_status"
    database_available = False
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
            database_available = True
        except Exception:  # pragma: no cover - depends on local integration services
            pytest.skip("PostgreSQL integration database is unavailable")

        await _seed_status_data(database, workspace_id=workspace_id)
        repository = CurrentClusterRepository(database)

        bootstrap = await repository.get_cluster_bootstrap_state(
            workspace_id=workspace_id,
            embedding_type="visual_style",
            model_name="status-model",
            dimension=4,
            milvus_collection="status-collection",
        )
        assert bootstrap.has_baseline is True
        assert bootstrap.run_in_progress is True
        assert bootstrap.eligible_asset_count == 5
        assert bootstrap.latest_run_id == "run_status_latest"
        assert bootstrap.latest_sample_count == 2

        status = await repository.get_new_asset_cluster_status(
            workspace_id=workspace_id,
            embedding_type="visual_style",
            model_name="status-model",
            dimension=4,
            milvus_collection="status-collection",
        )
        assert status.has_baseline is True
        assert status.baseline_cluster_run_id == "run_status_latest"
        assert status.baseline_sample_count == 2
        assert status.eligible_asset_count == 5
        assert {item.asset_id: item.status for item in status.items} == {
            "asset_status_reembedded": NewAssetClusterStatus.INCREMENTALLY_CLUSTERED,
            "asset_status_pending": NewAssetClusterStatus.PENDING,
            "asset_status_open": NewAssetClusterStatus.INCREMENTALLY_CLUSTERED,
            "asset_status_manual": NewAssetClusterStatus.MANUAL_MANAGEMENT,
        }
        by_id = {item.asset_id: item for item in status.items}
        assert by_id["asset_status_reembedded"].cluster_id == "cluster_status_dynamic"
        assert by_id["asset_status_reembedded"].score == pytest.approx(0.93)
        assert by_id["asset_status_pending"].cluster_id is None
        assert by_id["asset_status_manual"].cluster_name == "人工管理簇"

        async with database.session() as session, session.begin():
            await session.execute(
                delete(ClusterRun).where(
                    ClusterRun.workspace_id == workspace_id,
                    ClusterRun.status.in_(["completed", "insufficient_data"]),
                )
            )
        without_baseline = await repository.get_new_asset_cluster_status(
            workspace_id=workspace_id,
            embedding_type="visual_style",
            model_name="status-model",
            dimension=4,
            milvus_collection="status-collection",
        )
        assert without_baseline.has_baseline is False
        assert without_baseline.baseline_cluster_run_id is None
        assert without_baseline.baseline_sample_count is None
        assert without_baseline.eligible_asset_count == 5
        assert {item.asset_id for item in without_baseline.items} == {
            "asset_status_baseline",
            "asset_status_reembedded",
            "asset_status_pending",
            "asset_status_open",
            "asset_status_manual",
        }
    finally:
        try:
            if database_available:
                async with database.session() as session, session.begin():
                    await session.execute(
                        delete(Workspace).where(Workspace.workspace_id == workspace_id)
                    )
        finally:
            await database.dispose()


async def _seed_status_data(database: Database, *, workspace_id: str) -> None:
    now = datetime.now(UTC)
    async with database.session() as session, session.begin():
        session.add(Workspace(workspace_id=workspace_id, name="Status test"))
        await session.flush()
        session.add(
            SourceFile(
                source_file_id="src_new_asset_status",
                workspace_id=workspace_id,
                original_file_name="status.png",
                file_type=".png",
                mime_type="image/png",
                relative_path="status.png",
                file_tree_context=[],
                storage_uri="file:///tmp/status.png",
                sha256="1" * 64,
                processing_generation=1,
                file_size_bytes=1,
                processing_status="completed",
            )
        )
        await session.flush()

        eligible_ids = [
            "asset_status_baseline",
            "asset_status_reembedded",
            "asset_status_pending",
            "asset_status_open",
            "asset_status_manual",
        ]
        for index, asset_id in enumerate(eligible_ids):
            session.add(
                Asset(
                    asset_id=asset_id,
                    workspace_id=workspace_id,
                    source_file_id="src_new_asset_status",
                    asset_type="image",
                    file_name=f"status-{index}.png",
                    file_type=".png",
                    asset_key=f"status-{index}",
                    generation=1,
                    content_hash=f"{index + 10:064x}",
                    asset_name=f"状态素材 {index}",
                    asset_features={"visual_style": {"value": "摄影风格"}},
                    file_tree_context=[],
                    source_contexts=[],
                    file_info={},
                    source_locator={"type": "whole_file"},
                    processing_status="completed",
                    created_at=now + timedelta(seconds=index),
                )
            )
        session.add_all(
            [
                Asset(
                    asset_id="asset_status_ineligible",
                    workspace_id=workspace_id,
                    source_file_id="src_new_asset_status",
                    asset_type="image",
                    file_name="ineligible.png",
                    file_type=".png",
                    asset_key="ineligible",
                    generation=1,
                    content_hash="20" * 32,
                    asset_features={
                        "visual_style": {"value": None, "status": "not_applicable"}
                    },
                    file_tree_context=[],
                    source_contexts=[],
                    file_info={},
                    source_locator={},
                    processing_status="completed",
                ),
                Asset(
                    asset_id="asset_status_stale_generation",
                    workspace_id=workspace_id,
                    source_file_id="src_new_asset_status",
                    asset_type="image",
                    file_name="stale.png",
                    file_type=".png",
                    asset_key="stale",
                    generation=0,
                    content_hash="21" * 32,
                    asset_features={"visual_style": {"value": "过期风格"}},
                    file_tree_context=[],
                    source_contexts=[],
                    file_info={},
                    source_locator={},
                    processing_status="completed",
                ),
                Asset(
                    asset_id="asset_status_unsupported_type",
                    workspace_id=workspace_id,
                    source_file_id="src_new_asset_status",
                    asset_type="text_block",
                    file_name="unsupported.txt",
                    file_type=".txt",
                    asset_key="unsupported",
                    generation=1,
                    content_hash="22" * 32,
                    asset_features={"visual_style": {"value": "不应进入视觉通道"}},
                    file_tree_context=[],
                    source_contexts=[],
                    file_info={},
                    source_locator={},
                    processing_status="completed",
                ),
            ]
        )
        await session.flush()

        embedding_specs = [
            ("emb_status_baseline", "asset_status_baseline", now),
            ("emb_status_reembedded_old", "asset_status_reembedded", now),
            (
                "emb_status_reembedded_new",
                "asset_status_reembedded",
                now + timedelta(minutes=1),
            ),
            ("emb_status_pending", "asset_status_pending", now),
            ("emb_status_open", "asset_status_open", now),
            ("emb_status_manual", "asset_status_manual", now),
            ("emb_status_ineligible", "asset_status_ineligible", now),
            ("emb_status_stale", "asset_status_stale_generation", now),
            ("emb_status_unsupported", "asset_status_unsupported_type", now),
        ]
        for index, (embedding_id, asset_id, created_at) in enumerate(embedding_specs):
            session.add(
                EmbeddingRecord(
                    embedding_id=embedding_id,
                    workspace_id=workspace_id,
                    asset_id=asset_id,
                    embedding_type="visual_style",
                    model_name="status-model",
                    dimension=4,
                    source_content_hash=f"{index + 30:064x}",
                    embedding_source_mode="feature_text",
                    milvus_collection="status-collection",
                    milvus_primary_key=embedding_id,
                    status="indexed",
                    created_at=created_at,
                )
            )
        session.add_all(
            [
                ClusterRun(
                    cluster_run_id="run_status_older",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    input_embedding_ids=[item[0] for item in embedding_specs],
                    dataset_hash="2" * 64,
                    sample_count=len(embedding_specs),
                    preprocessing={},
                    parameters={},
                    status="completed",
                    started_at=now - timedelta(hours=4),
                    completed_at=now - timedelta(hours=3),
                ),
                ClusterRun(
                    cluster_run_id="run_status_latest",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    input_embedding_ids=[
                        "emb_status_baseline",
                        "emb_status_reembedded_old",
                    ],
                    dataset_hash="3" * 64,
                    sample_count=2,
                    preprocessing={},
                    parameters={},
                    status="insufficient_data",
                    started_at=now - timedelta(hours=2),
                    completed_at=now - timedelta(hours=1),
                ),
                ClusterRun(
                    cluster_run_id="run_status_pending",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    input_embedding_ids=[],
                    dataset_hash="0" * 64,
                    sample_count=0,
                    preprocessing={"trigger": "bootstrap"},
                    parameters={},
                    status="pending",
                ),
            ]
        )
        session.add_all(
            [
                CurrentCluster(
                    cluster_id="cluster_status_dynamic",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    mode="dynamic",
                    name="动态簇",
                    description="自动增量聚类。",
                ),
                CurrentCluster(
                    cluster_id="cluster_status_open",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    mode="resident_open",
                    name="开放常驻簇",
                    description="允许增量进入。",
                ),
                CurrentCluster(
                    cluster_id="cluster_status_manual",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    mode="resident_manual",
                    name="人工管理簇",
                    description="仅由用户管理。",
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                CurrentClusterMember(
                    cluster_id="cluster_status_dynamic",
                    asset_id="asset_status_reembedded",
                    embedding_type="visual_style",
                    source="incremental",
                    score=0.93,
                ),
                CurrentClusterMember(
                    cluster_id="cluster_status_open",
                    asset_id="asset_status_open",
                    embedding_type="visual_style",
                    source="incremental",
                    score=0.88,
                ),
                CurrentClusterMember(
                    cluster_id="cluster_status_manual",
                    asset_id="asset_status_manual",
                    embedding_type="visual_style",
                    source="user",
                    score=None,
                ),
            ]
        )
