from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import Asset, SourceFile, Workspace
from capsule.db.repositories import RelationGraphRepository
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
