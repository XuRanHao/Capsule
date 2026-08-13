from datetime import UTC, datetime

from capsule.db.repositories import EmbeddingAsset
from capsule.relation_graph import (
    AssetEntityRelationResolution,
    apply_asset_entity_relations,
    build_relation_graph,
)
from capsule.schemas import AssetUnderstanding


def _asset(asset_id: str, path: str) -> EmbeddingAsset:
    return EmbeddingAsset(
        asset_id=asset_id,
        workspace_id="workspace",
        project_id="project",
        source_file_id=f"source_{asset_id}",
        asset_type="image",
        file_type=".png",
        content_hash=asset_id * 8,
        embedding_revision=1,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri="file:///tmp/test.png",
        source_mime_type="image/png",
        source_relative_path=path,
    )


def _understanding(subject: str, description: str) -> AssetUnderstanding:
    return AssetUnderstanding.model_validate(
        {
            "asset_name": subject,
            "asset_description": description,
            "features": {
                "subject_content": {
                    "applicability": "applicable",
                    "items": [
                        {
                            "subject": subject,
                            "description": description,
                            "salience": 0.95,
                            "status": "metadata",
                            "evidence": [],
                            "ocr_confidence": None,
                        }
                    ],
                }
            },
        }
    )


def test_graph_merges_metadata_and_content_entities_then_links_assets() -> None:
    assets = [
        _asset("asset_a", "封不觉/全身.png"),
        _asset("asset_b", "封不觉/近景.png"),
        _asset("asset_c", "汪叹之/全身.png"),
    ]
    graph = build_relation_graph(
        assets,
        {
            "asset_a": _understanding("封不觉", "黑发红眼、身穿紫色长外套的男性角色"),
            "asset_b": _understanding("封不觉", "黑发红眼男性角色的正面近景"),
            "asset_c": _understanding("汪叹之", "金发蓝眼、身穿白色斗篷的男性角色"),
        },
        metadata_content_relations={"封不觉": {"relation": "same_entity"}},
    )

    assert graph["asset_count"] == 3
    assert graph["entity_count"] == 1
    same_entity_edges = [edge for edge in graph["edges"] if edge["relation"] == "SAME_ENTITY"]
    assert same_entity_edges == [
        {
            "source": "asset_a",
            "target": "asset_b",
            "relation": "SAME_ENTITY",
            "description": "两项素材共同关联实体封不觉",
        }
    ]
    feng = next(entity for entity in graph["entities"] if entity["name"] == "封不觉")
    assert feng["origins"] == ["content", "metadata"]
    assert feng["semantic"] == "黑发红眼、身穿紫色长外套的男性角色"
    apply_asset_entity_relations(
        graph,
        AssetEntityRelationResolution.model_validate(
            {
                "relations": [
                    {
                        "source_id": "asset_a",
                        "target_id": feng["entity_id"],
                        "establishes_relation": True,
                        "relation": "DEPICTS",
                        "description": "画面描绘封不觉。",
                    },
                    {
                        "source_id": "asset_b",
                        "target_id": feng["entity_id"],
                        "establishes_relation": True,
                        "relation": "DEPICTS",
                        "description": "近景描绘封不觉。",
                    },
                ]
            }
        ),
    )
    membership = [edge for edge in graph["edges"] if edge["source"] == "asset_a"]
    assert membership[0]["relation"] == "DEPICTS"
    assert membership[0]["description"].startswith("画面描绘封不觉")


def test_graph_keeps_single_asset_subjects_out_of_virtual_nodes() -> None:
    assets = [
        _asset("asset_a", "飞船内部/中央大厅.png"),
        _asset("asset_b", "飞船内部/宴会大厅.png"),
    ]
    graph = build_relation_graph(
        assets,
        {
            "asset_a": _understanding("中央雕塑", "飞船中央大厅里的大型人像雕塑"),
            "asset_b": _understanding("宴会厅", "带长桌与吊灯的宴会空间"),
        },
        metadata_content_relations={"飞船内部": "contains_content"},
    )

    assert {entity["name"] for entity in graph["entities"]} == {"飞船内部"}
    assert [item["subject"] for item in graph["assets"][0]["main_entities"]] == [
        "飞船内部",
    ]
    assert graph["assets"][0]["primary_subject"]["subject"] == "中央雕塑"
    shared = [edge for edge in graph["edges"] if edge["relation"] == "SHARES_METADATA_ENTITY"]
    assert len(shared) == 1
    assert shared[0]["description"] == "两项素材共同关联实体飞船内部"
    assert not [edge for edge in graph["edges"] if edge["relation"] == "CONTAINS"]
    assert all(len(entity["asset_ids"]) >= 2 for entity in graph["entities"])


def test_graph_rejects_storage_only_metadata_entity() -> None:
    assets = [
        _asset("asset_a", "测试素材/a.png"),
        _asset("asset_b", "测试素材/b.png"),
    ]
    graph = build_relation_graph(
        assets,
        {
            "asset_a": _understanding("人物", "一名人物"),
            "asset_b": _understanding("建筑", "一栋建筑"),
        },
        metadata_content_relations={
            "测试素材": {
                "relation": "contains_content",
                "entity_semantic": "用于存放测试文件的通用容器",
                "build_entity": False,
            }
        },
    )

    assert graph["entities"] == []
    assert graph["edges"] == []


def test_agent_rejection_removes_edge_and_asset_pair_cliques() -> None:
    assets = [
        _asset("asset_a", "飞船内部/大厅.png"),
        _asset("asset_b", "飞船内部/雕塑.png"),
        _asset("asset_c", "飞船内部/神社.png"),
    ]
    graph = build_relation_graph(
        assets,
        {
            "asset_a": _understanding("大厅", "飞船大厅"),
            "asset_b": _understanding("雕塑", "大厅中央雕塑"),
            "asset_c": _understanding("男性", "神社前的男性"),
        },
        metadata_content_relations={"飞船内部": "contains_content"},
    )
    entity = graph["entities"][0]
    apply_asset_entity_relations(
        graph,
        AssetEntityRelationResolution.model_validate(
            {
                "relations": [
                    {
                        "source_id": "asset_a",
                        "target_id": entity["entity_id"],
                        "establishes_relation": True,
                        "relation": "位于",
                        "description": "大厅位于飞船内部",
                    },
                    {
                        "source_id": "asset_b",
                        "target_id": entity["entity_id"],
                        "establishes_relation": True,
                        "relation": "位于",
                        "description": "雕塑位于飞船内部",
                    },
                    {
                        "source_id": "asset_c",
                        "target_id": entity["entity_id"],
                        "establishes_relation": False,
                        "relation": "不相关",
                        "description": "神社与飞船内部无关",
                    },
                ]
            }
        ),
    )

    assert {edge["source"] for edge in graph["edges"]} == {"asset_a", "asset_b"}
    assert graph["edge_count"] == 2
    assert graph["rejected_relations"][0]["description"] == "神社与飞船内部无关"


def test_asset_entity_candidate_pairs_are_deduplicated_before_persistence() -> None:
    graph = {
        "assets": [],
        "entities": [
            {
                "entity_id": "entity_a",
                "name": "实体A",
                "semantic": "实体A",
                "asset_ids": [],
            }
        ],
        "edges": [
            {"source": "asset_a", "target": "entity_a", "relation": "CANDIDATE"},
            {"source": "asset_a", "target": "entity_a", "relation": "CANDIDATE"},
        ],
    }

    apply_asset_entity_relations(
        graph,
        AssetEntityRelationResolution.model_validate(
            {
                "relations": [
                    {
                        "source_id": "asset_a",
                        "target_id": "entity_a",
                        "establishes_relation": False,
                        "relation": "",
                        "description": "语义不匹配",
                    }
                ]
            }
        ),
    )

    assert graph["edges"] == []
    assert graph["rejected_relations"] == [
        {
            "source_id": "asset_a",
            "target_id": "entity_a",
            "establishes_relation": False,
            "relation": "",
            "description": "语义不匹配",
        }
    ]
