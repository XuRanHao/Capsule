from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, SourceFile, Workspace
from capsule.db.repositories import (
    HierarchyBatchCommit,
    HierarchyCandidateSync,
    HierarchyEntityCandidate,
    HierarchyEntityWrite,
    RelationGraphRepository,
)
from capsule.db.session import Database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_relation_graph_repository_round_trips_entity_edges() -> None:
    database = Database(get_settings())
    suffix = uuid4().hex[:12]
    workspace_id = f"workspace_relation_graph_{suffix}"
    asset_id = f"asset_relation_graph_{suffix}"
    entity_a = f"entity_a_{suffix}"
    entity_b = f"entity_b_{suffix}"
    database_available = False
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
            database_available = True
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        await _seed_asset(database, workspace_id=workspace_id, asset_id=asset_id)
        repository = RelationGraphRepository(database)
        graph = {
            "entities": [
                {
                    "entity_id": entity_a,
                    "name": "A人物武器",
                    "semantic": "A人物使用的武器设定",
                },
                {
                    "entity_id": entity_b,
                    "name": "A人物",
                    "semantic": "角色A及其人物设定",
                },
            ],
            "edges": [
                {
                    "source": asset_id,
                    "target": entity_a,
                    "relation": "CLUSTER_MEMBER",
                    "description": "内容高度相似",
                }
            ],
            "entity_edges": [
                {
                    "source_entity_id": entity_a,
                    "target_entity_id": entity_b,
                    "relation": "WEAPON_OF",
                    "description": "该节点是A人物的武器设定。",
                    "edge_type": "hierarchy",
                }
            ],
        }

        assert await repository.replace(
            workspace_id=workspace_id,
            input_revision="1" * 64,
            graph=graph,
            candidates=[],
            subject_cluster_status={"status": "ready"},
            asset_revisions={asset_id: "2" * 64},
        ) == 1

        loaded = await repository.load(
            workspace_id=workspace_id,
            input_revision="1" * 64,
        )
        assert loaded is not None
        assert loaded["entity_edges"] == [
            {
                "source_entity_id": entity_a,
                "target_entity_id": entity_b,
                "relation": "WEAPON_OF",
                "description": "该节点是A人物的武器设定。",
                "edge_type": "hierarchy",
            }
        ]

        graph_without_entity_edges = {**graph, "edges": []}
        graph_without_entity_edges.pop("entity_edges")
        assert await repository.replace(
            workspace_id=workspace_id,
            input_revision="3" * 64,
            graph=graph_without_entity_edges,
            candidates=[],
            subject_cluster_status={"status": "ready"},
            asset_revisions={asset_id: "4" * 64},
        ) == 2
        reloaded = await repository.load_current(workspace_id=workspace_id)
        assert reloaded is not None
        assert reloaded["entity_edges"] == []
    finally:
        try:
            if database_available:
                async with database.session() as session, session.begin():
                    await session.execute(
                        delete(Workspace).where(Workspace.workspace_id == workspace_id)
                    )
        finally:
            await database.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hierarchy_batch_commit_is_idempotent_and_enforces_single_asset_owner() -> None:
    database = Database(get_settings())
    suffix = uuid4().hex[:12]
    workspace_id = f"workspace_hierarchy_commit_{suffix}"
    asset_id = f"asset_hierarchy_commit_{suffix}"
    database_available = False
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
            database_available = True
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        await _seed_asset(database, workspace_id=workspace_id, asset_id=asset_id)
        repository = RelationGraphRepository(database)
        initialization = await repository.initialize_hierarchy_graph(
            workspace_id=workspace_id,
            input_revision="0" * 64,
            subject_cluster_status={"status": "ready"},
        )
        sync = await repository.sync_hierarchy_entity_candidates(
            workspace_id=workspace_id,
            sync=HierarchyCandidateSync(
                operation_id=f"sync_{suffix}",
                expected_build_version=initialization.build_version,
                expected_input_revision="0" * 64,
                input_revision="1" * 64,
                subject_cluster_status={"status": "ready"},
                candidates=(
                    HierarchyEntityCandidate(
                        candidate_id=f"subject_cluster:{suffix}",
                        origin="subject_cluster",
                        name="角色A",
                        semantic="角色A相关设定",
                        asset_ids=(asset_id,),
                        embedding_vector=(1.0, 0.0),
                        embedding_model="test-embedding",
                    ),
                ),
                governed_candidate_ids=(f"subject_cluster:{suffix}",),
                asset_revisions={asset_id: "2" * 64},
            ),
        )
        entity_id = sync.pending_entity_ids[0]
        commit = HierarchyBatchCommit(
            operation_id=f"group_{suffix}",
            expected_build_version=sync.build_version,
            expected_input_revision=sync.input_revision,
            input_revision="3" * 64,
            subject_cluster_status={"status": "ready"},
            operations=(
                {
                    "type": "group",
                    "parent": {
                        "mode": "create",
                        "temporary_parent_id": "virtual:character_setting",
                        "name": "角色A设定",
                        "semantic": "角色A的完整设定集合",
                    },
                    "children": [
                        {
                            "child_entity_id": entity_id,
                            "relation": "角色设定",
                            "description": "该实体是角色A设定的一部分。",
                        }
                    ],
                },
            ),
            entity_writes=(
                HierarchyEntityWrite(
                    entity_id=f"entity_parent_{suffix}",
                    name="角色A设定",
                    semantic="角色A的完整设定集合",
                    origins=("agent_structure",),
                    descriptions=("角色A的完整设定集合",),
                    embedding_vector=(0.5, 0.5),
                    embedding_model="test-embedding",
                    temporary_parent_id="virtual:character_setting",
                ),
            ),
        )
        first = await repository.commit_hierarchy_batch(
            workspace_id=workspace_id,
            commit=commit,
        )
        replay = await repository.commit_hierarchy_batch(
            workspace_id=workspace_id,
            commit=commit,
        )
        assert first.replayed is False
        assert replay.replayed is True
        assert replay.build_version == first.build_version

        graph = await repository.load_current(workspace_id=workspace_id)
        assert graph is not None
        assert graph["build_version"] == first.build_version
        assert graph["input_revision"] == "3" * 64
        assert graph["edges"] == [
            {
                "source": asset_id,
                "target": entity_id,
                "relation": "CLUSTER_MEMBER",
                "description": "受治理主体簇成员。",
                "content_subject": "",
            }
        ]
        assert graph["entity_edges"] == [
            {
                "source_entity_id": entity_id,
                "target_entity_id": f"entity_parent_{suffix}",
                "relation": "角色设定",
                "description": "该实体是角色A设定的一部分。",
                "edge_type": "hierarchy",
            }
        ]
    finally:
        try:
            if database_available:
                async with database.session() as session, session.begin():
                    await session.execute(
                        delete(Workspace).where(Workspace.workspace_id == workspace_id)
                    )
        finally:
            await database.dispose()


async def _seed_asset(database: Database, *, workspace_id: str, asset_id: str) -> None:
    source_file_id = f"source_{asset_id}"
    async with database.session() as session, session.begin():
        session.add(Workspace(workspace_id=workspace_id, name="Relation graph test"))
        await session.flush()
        session.add(
            SourceFile(
                source_file_id=source_file_id,
                workspace_id=workspace_id,
                original_file_name="character.png",
                file_type=".png",
                mime_type="image/png",
                relative_path="NPC/A人物/character.png",
                file_tree_context=["NPC", "A人物"],
                storage_uri="file:///tmp/character.png",
                sha256="5" * 64,
                file_size_bytes=4,
                processing_status="completed",
            )
        )
        await session.flush()
        session.add(
            Asset(
                asset_id=asset_id,
                workspace_id=workspace_id,
                source_file_id=source_file_id,
                asset_type="image",
                file_name="character.png",
                file_type=".png",
                asset_key="whole-file",
                content_hash="6" * 64,
                asset_description="A人物的武器设定图。",
                asset_features={},
                file_tree_context=["NPC", "A人物"],
                source_contexts=[],
                file_info={},
                source_locator={"type": "whole_file"},
                processing_status="completed",
            )
        )
