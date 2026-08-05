import pytest
from sqlalchemy import delete, text

from capsule.config import get_settings
from capsule.db.models import Asset, ClusterRun, EmbeddingRecord, SourceFile, Workspace
from capsule.db.repositories import (
    CurrentClusterMemberWrite,
    CurrentClusterPublish,
    CurrentClusterRepository,
)
from capsule.db.session import Database
from capsule.enums import ClusterMemberSource, ClusterMode


@pytest.mark.integration
@pytest.mark.asyncio
async def test_current_clusters_preserve_residents_and_honor_manual_membership() -> None:
    database = Database(get_settings())
    workspace_id = "workspace_current_cluster_repository"
    database_available = False
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
            database_available = True
        except Exception:  # pragma: no cover - depends on the local integration services
            pytest.skip("PostgreSQL integration database is unavailable")

        await _seed_cluster_data(database, workspace_id=workspace_id)
        repository = CurrentClusterRepository(database)
        first_publish = await repository.publish_dynamic_clusters(
            run_id="run_current_cluster_1",
            workspace_id=workspace_id,
            embedding_type="visual_style",
            clusters=[
                CurrentClusterPublish(
                    cluster_id="cluster_current_resident",
                    name="霓虹夜景",
                    description="以霓虹和城市夜景为共同视觉风格。",
                    representative_asset_id="asset_current_cluster_0",
                    members=[
                        CurrentClusterMemberWrite("asset_current_cluster_0", 0.98),
                        CurrentClusterMemberWrite("asset_current_cluster_1", 0.91),
                    ],
                )
            ],
        )
        assert [cluster.cluster_id for cluster in first_publish] == [
            "cluster_current_resident"
        ]

        resident = await repository.set_mode(
            cluster_id="cluster_current_resident",
            workspace_id=workspace_id,
            mode=ClusterMode.RESIDENT_OPEN,
        )
        assert resident.mode is ClusterMode.RESIDENT_OPEN
        attached = await repository.attach_members(
            cluster_id=resident.cluster_id,
            workspace_id=workspace_id,
            asset_ids=["asset_current_cluster_2"],
            source=ClusterMemberSource.USER,
        )
        assert attached[0].source is ClusterMemberSource.USER

        detached = await repository.detach_members(
            cluster_id=resident.cluster_id,
            workspace_id=workspace_id,
            asset_ids=["asset_current_cluster_1"],
            created_by="user_test",
        )
        assert detached == ["asset_current_cluster_1"]
        assert {
            (item.cluster_id, item.asset_id)
            for item in await repository.list_exclusions(
                cluster_id=resident.cluster_id,
                workspace_id=workspace_id,
            )
        } == {(resident.cluster_id, "asset_current_cluster_1")}
        assert await repository.list_excluded_pairs(
            workspace_id=workspace_id,
            embedding_type="visual_style",
            cluster_ids=[resident.cluster_id],
            asset_ids=["asset_current_cluster_1", "asset_current_cluster_3"],
        ) == {(resident.cluster_id, "asset_current_cluster_1")}
        indexed = await repository.list_indexed_asset_embeddings(
            workspace_id=workspace_id,
            embedding_type="visual_style",
            asset_ids=["asset_current_cluster_1"],
        )
        assert [(item.asset_id, item.embedding_id) for item in indexed] == [
            ("asset_current_cluster_1", "emb_current_cluster_1")
        ]

        # An algorithmic retry may not undo the user's removal.
        blocked = await repository.attach_members(
            cluster_id=resident.cluster_id,
            workspace_id=workspace_id,
            asset_ids=["asset_current_cluster_1"],
            source=ClusterMemberSource.INCREMENTAL,
            scores={"asset_current_cluster_1": 0.99},
        )
        assert blocked == []

        restored = await repository.attach_members(
            cluster_id=resident.cluster_id,
            workspace_id=workspace_id,
            asset_ids=["asset_current_cluster_1"],
            source=ClusterMemberSource.USER,
        )
        assert [item.asset_id for item in restored] == ["asset_current_cluster_1"]
        assert await repository.list_exclusions(
            cluster_id=resident.cluster_id,
            workspace_id=workspace_id,
        ) == []

        await repository.publish_dynamic_clusters(
            run_id="run_current_cluster_2",
            workspace_id=workspace_id,
            embedding_type="visual_style",
            clusters=[
                CurrentClusterPublish(
                    cluster_id="cluster_current_dynamic_2",
                    name="自然风景",
                    description="下一次运行生成的动态自然风景簇。",
                    representative_asset_id="asset_current_cluster_3",
                    members=[CurrentClusterMemberWrite("asset_current_cluster_3", 0.94)],
                )
            ],
        )
        current = await repository.list_clusters(
            workspace_id=workspace_id,
            embedding_type="visual_style",
        )
        assert {(item.cluster_id, item.mode) for item in current} == {
            ("cluster_current_resident", ClusterMode.RESIDENT_OPEN),
            ("cluster_current_dynamic_2", ClusterMode.DYNAMIC),
        }
        assert await repository.list_resident_asset_ids(
            workspace_id=workspace_id,
            embedding_type="visual_style",
        ) == {
            "asset_current_cluster_0",
            "asset_current_cluster_1",
            "asset_current_cluster_2",
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


async def _seed_cluster_data(database: Database, *, workspace_id: str) -> None:
    async with database.session() as session, session.begin():
        session.add(Workspace(workspace_id=workspace_id, name="Current cluster test"))
        await session.flush()
        session.add(
            SourceFile(
                source_file_id="src_current_cluster",
                workspace_id=workspace_id,
                original_file_name="cluster.png",
                file_type=".png",
                mime_type="image/png",
                relative_path="cluster.png",
                file_tree_context=[],
                storage_uri="file:///tmp/cluster.png",
                sha256="1" * 64,
                file_size_bytes=4,
                processing_status="completed",
            )
        )
        await session.flush()
        for index in range(4):
            asset_id = f"asset_current_cluster_{index}"
            session.add(
                Asset(
                    asset_id=asset_id,
                    workspace_id=workspace_id,
                    source_file_id="src_current_cluster",
                    asset_type="image",
                    file_name=f"cluster-{index}.png",
                    file_type=".png",
                    asset_key=f"cluster-{index}",
                    content_hash=f"{index + 10:064x}",
                    asset_description="聚类仓储测试素材。",
                    asset_features={"visual_style": {"value": "测试风格"}},
                    file_tree_context=[],
                    source_contexts=[],
                    file_info={},
                    source_locator={"type": "whole_file"},
                    processing_status="completed",
                )
            )
        await session.flush()
        for index in range(4):
            asset_id = f"asset_current_cluster_{index}"
            session.add(
                EmbeddingRecord(
                    embedding_id=f"emb_current_cluster_{index}",
                    workspace_id=workspace_id,
                    asset_id=asset_id,
                    embedding_type="visual_style",
                    model_name="test-model",
                    dimension=4,
                    source_content_hash=f"{index + 20:064x}",
                    embedding_source_mode="feature_text",
                    milvus_collection="test-clusters",
                    milvus_primary_key=f"emb_current_cluster_{index}",
                    status="indexed",
                )
            )
        for run_number in (1, 2):
            session.add(
                ClusterRun(
                    cluster_run_id=f"run_current_cluster_{run_number}",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    input_embedding_ids=[
                        f"emb_current_cluster_{index}" for index in range(4)
                    ],
                    dataset_hash=str(run_number) * 64,
                    sample_count=4,
                    preprocessing={},
                    parameters={},
                    cluster_count=1,
                    noise_count=0,
                    noise_ratio=0.0,
                    status="completed",
                )
            )
