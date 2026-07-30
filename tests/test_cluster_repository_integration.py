import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import (
    Asset,
    ClusterCapsule,
    ClusterRepresentativeAsset,
    ClusterRun,
    SourceFile,
    Workspace,
)
from capsule.db.repositories import ClusterRepository
from capsule.db.session import Database
from capsule.enums import ClusterInternalVariance, ClusterRepresentativeRole
from capsule.schemas import ClusterCapsuleWrite, ClusterRepresentativeWrite, ClusterSummary


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cluster_capsule_persists_asset_references_and_user_overrides() -> None:
    database = Database(get_settings())
    workspace_id = "workspace_cluster_repository_integration"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        async with database.session() as session, session.begin():
            session.add(Workspace(workspace_id=workspace_id, name="Cluster test"))
            await session.flush()
            for index in range(3):
                source_file_id = f"src_cluster_test_{index}"
                session.add(
                    SourceFile(
                        source_file_id=source_file_id,
                        workspace_id=workspace_id,
                        original_file_name=f"image-{index}.png",
                        file_type=".png",
                        mime_type="image/png",
                        relative_path=f"images/image-{index}.png",
                        file_tree_context=["images"],
                        storage_uri=f"file:///tmp/image-{index}.png",
                        sha256=f"{index:064x}",
                        file_size_bytes=1,
                        processing_status="completed",
                    )
                )
            await session.flush()
            for index in range(3):
                source_file_id = f"src_cluster_test_{index}"
                session.add(
                    Asset(
                        asset_id=f"asset_cluster_test_{index}",
                        workspace_id=workspace_id,
                        source_file_id=source_file_id,
                        asset_type="image",
                        file_name=f"image-{index}.png",
                        file_type=".png",
                        asset_key=f"image-{index}",
                        content_hash=f"{index + 10:064x}",
                        asset_description=f"第 {index} 张测试图片。",
                        asset_features={"visual_style": {"value": "霓虹"}},
                        file_tree_context=["images"],
                        source_contexts=[],
                        file_info={},
                        source_locator={"type": "whole_file"},
                        processing_status="completed",
                    )
                )
            session.add(
                ClusterRun(
                    cluster_run_id="run_cluster_test",
                    workspace_id=workspace_id,
                    embedding_type="visual_style",
                    input_embedding_ids=["emb_1", "emb_2", "emb_3"],
                    dataset_hash="a" * 64,
                    sample_count=20,
                    preprocessing={"pca_dimension": 64},
                    parameters={"min_cluster_size": 3},
                    cluster_count=1,
                    noise_count=0,
                    noise_ratio=0.0,
                    status="completed",
                )
            )

        repository = ClusterRepository(database)
        stored = await repository.upsert_capsule(
            _capsule_write(
                workspace_id=workspace_id,
                name="蓝紫色霓虹夜景",
                description=_description("蓝紫色霓虹夜景"),
            )
        )

        assert stored.medoid_asset_id == "asset_cluster_test_0"
        assert stored.representative_asset_ids == [
            "asset_cluster_test_0",
            "asset_cluster_test_1",
            "asset_cluster_test_2",
        ]
        async with database.session() as session:
            capsule = await session.get(ClusterCapsule, stored.cluster_capsule_id)
            representatives = list(
                await session.scalars(
                    select(ClusterRepresentativeAsset)
                    .where(
                        ClusterRepresentativeAsset.cluster_capsule_id == stored.cluster_capsule_id
                    )
                    .order_by(ClusterRepresentativeAsset.rank)
                )
            )
            assert capsule is not None
            assert [item.asset_id for item in representatives] == stored.representative_asset_ids
            assert representatives[0].role == ClusterRepresentativeRole.MEDOID.value

        await repository.set_name_override(
            cluster_capsule_id=stored.cluster_capsule_id,
            workspace_id=workspace_id,
            name="人工命名",
        )
        await repository.set_description_override(
            cluster_capsule_id=stored.cluster_capsule_id,
            workspace_id=workspace_id,
            description="人工说明：该组由用户确认用于夜景灵感整理。",
        )
        regenerated = await repository.upsert_capsule(
            _capsule_write(
                workspace_id=workspace_id,
                name="模型重新命名",
                description=_description("模型重新命名"),
            )
        )

        assert regenerated.model_generated_name == "模型重新命名"
        assert regenerated.effective_name == "人工命名"
        assert regenerated.effective_description.startswith("人工说明")
        restored_name = await repository.set_name_override(
            cluster_capsule_id=stored.cluster_capsule_id,
            workspace_id=workspace_id,
            name=None,
        )
        restored_description = await repository.set_description_override(
            cluster_capsule_id=stored.cluster_capsule_id,
            workspace_id=workspace_id,
            description=None,
        )
        assert restored_name.effective_name == "模型重新命名"
        assert restored_description.effective_description == _description("模型重新命名")
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(Workspace.workspace_id == workspace_id)
                )
        finally:
            await database.dispose()


def _capsule_write(*, workspace_id: str, name: str, description: str) -> ClusterCapsuleWrite:
    return ClusterCapsuleWrite(
        cluster_run_id="run_cluster_test",
        workspace_id=workspace_id,
        embedding_type="visual_style",
        cluster_label=0,
        summary=ClusterSummary(
            name=name,
            description=description,
            keywords=["霓虹", "夜景", "赛博朋克"],
            common_features=["蓝紫色冷光", "城市夜景"],
            internal_variance=ClusterInternalVariance.LOW,
        ),
        member_count=20,
        average_membership_probability=0.87,
        representatives=[
            ClusterRepresentativeWrite(
                asset_id="asset_cluster_test_0",
                role=ClusterRepresentativeRole.MEDOID,
                rank=0,
                distance_to_medoid=0.0,
                membership_probability=0.99,
            ),
            ClusterRepresentativeWrite(
                asset_id="asset_cluster_test_1",
                role=ClusterRepresentativeRole.CORE,
                rank=1,
                distance_to_medoid=0.1,
                membership_probability=0.92,
            ),
            ClusterRepresentativeWrite(
                asset_id="asset_cluster_test_2",
                role=ClusterRepresentativeRole.CORE,
                rank=2,
                distance_to_medoid=0.2,
                membership_probability=0.88,
            ),
        ],
    )


def _description(name: str) -> str:
    return (
        f"{name} 这一组素材以蓝紫色霓虹光影和城市夜景为主要共同特征，"
        "画面多使用冷色调反射与高对比照明，整体呈现稳定的电影感视觉风格。"
    )
