import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoClient, DoubaoResponseError
from capsule.relation_graph import (
    CreatedGroupParent,
    EntityStructureOperationResolution,
    GroupEntityOperation,
    MergeEntityOperation,
    SeparateEntityOperation,
    merge_entities_with_high_asset_overlap,
)


def _overlap_graph(*, second_asset_ids: list[str]) -> dict[str, Any]:
    asset_ids = ["asset_1", "asset_2", "asset_3", "asset_4"]
    return {
        "assets": [{"asset_id": asset_id} for asset_id in asset_ids],
        "entities": [
            {
                "entity_id": "entity_woody_material",
                "name": "伍迪角色素材",
                "semantic": "伍迪角色的素材",
                "asset_ids": asset_ids,
                "origins": ["subject_cluster"],
                "descriptions": [],
                "candidate_ids": ["candidate_1"],
            },
            {
                "entity_id": "entity_woody_asset",
                "name": "伍迪角色资产",
                "semantic": "伍迪角色的资产",
                "asset_ids": second_asset_ids,
                "origins": ["subject_cluster"],
                "descriptions": [],
                "candidate_ids": ["candidate_2"],
            },
            {
                "entity_id": "entity_woody",
                "name": "伍迪",
                "semantic": "伍迪角色",
                "asset_ids": [],
                "origins": ["agent_structure"],
                "descriptions": [],
                "candidate_ids": [],
            },
        ],
        "edges": [
            {
                "source": asset_id,
                "target": target,
                "relation": "RELATED",
                "description": "属于伍迪角色",
            }
            for asset_id in asset_ids
            for target in ("entity_woody_material", "entity_woody_asset")
            if target == "entity_woody_material" or asset_id in second_asset_ids
        ],
        "entity_edges": [
            {
                "source_entity_id": child_id,
                "target_entity_id": "entity_woody",
                "relation": "属于",
                "description": "伍迪角色的素材",
                "edge_type": "hierarchy",
            }
            for child_id in ("entity_woody_material", "entity_woody_asset")
        ],
        "rejected_relations": [
            {
                "source_id": "asset_other",
                "target_id": "entity_woody_asset",
                "establishes_relation": False,
                "relation": "",
                "description": "无关",
            }
        ],
    }


def test_high_asset_overlap_merges_entities_and_collapses_single_child_parent() -> None:
    graph = _overlap_graph(
        second_asset_ids=["asset_1", "asset_2", "asset_3", "asset_4"]
    )

    merged_count = merge_entities_with_high_asset_overlap(graph)

    assert merged_count == 1
    assert [entity["entity_id"] for entity in graph["entities"]] == [
        "entity_woody_material"
    ]
    assert graph["entity_edges"] == []
    assert len(graph["edges"]) == 4
    assert {edge["target"] for edge in graph["edges"]} == {
        "entity_woody_material"
    }
    assert graph["rejected_relations"][0]["target_id"] == "entity_woody_material"


def test_asset_overlap_below_threshold_keeps_entities_separate() -> None:
    graph = _overlap_graph(second_asset_ids=["asset_1", "asset_2"])

    merged_count = merge_entities_with_high_asset_overlap(graph)

    assert merged_count == 0
    assert len(graph["entities"]) == 3
    assert len(graph["entity_edges"]) == 2


def test_entity_structure_operations_parse_strict_discriminated_protocol() -> None:
    resolution = EntityStructureOperationResolution.model_validate(
        {
            "operations": [
                {
                    "type": "merge",
                    "source_entity_ids": ["entity_a", "entity_a_alias"],
                    "canonical_entity_id": "entity_a",
                    "name": "A人物",
                    "semantic": "项目中的角色A",
                },
                {"type": "separate", "entity_ids": ["entity_scene"]},
                {
                    "type": "group",
                    "parent": {
                        "mode": "create",
                        "temporary_parent_id": "virtual:a_settings",
                        "name": "A人物设定",
                        "semantic": "角色A相关的外观和装备设定",
                    },
                    "children": [
                        {
                            "child_entity_id": "entity_weapon",
                            "relation": "角色武器",
                            "description": "该节点描述A人物使用的武器。",
                        },
                        {
                            "child_entity_id": "entity_costume",
                            "relation": "服装设定",
                            "description": "该节点描述A人物的服装设计。",
                        },
                    ],
                },
            ]
        }
    )

    assert isinstance(resolution.operations[0], MergeEntityOperation)
    assert isinstance(resolution.operations[1], SeparateEntityOperation)
    group = resolution.operations[2]
    assert isinstance(group, GroupEntityOperation)
    assert isinstance(group.parent, CreatedGroupParent)


def test_entity_structure_operations_forbid_reason_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="reason"):
        EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "separate",
                        "entity_ids": ["entity_a"],
                        "reason": "不相似",
                    }
                ]
            }
        )


def test_new_group_parent_with_one_child_can_be_parsed_for_client_normalization() -> None:
    resolution = EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "group",
                        "parent": {
                            "mode": "create",
                            "temporary_parent_id": "virtual:a_settings",
                            "name": "A人物设定",
                            "semantic": "角色A相关设定",
                        },
                        "children": [
                            {
                                "child_entity_id": "entity_weapon",
                                "relation": "角色武器",
                                "description": "A人物的武器。",
                            }
                        ],
                    }
                ]
            }
        )
    assert isinstance(resolution.operations[0], GroupEntityOperation)


@pytest.mark.asyncio
async def test_entity_structure_client_sends_complete_context_and_returns_operations() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.update(payload)
        content = {
            "operations": [
                {
                    "type": "group",
                    "parent": {"mode": "reuse", "parent_entity_id": "entity_a"},
                    "children": [
                        {
                            "child_entity_id": "entity_weapon",
                            "relation": "角色武器",
                            "description": "A人物使用的武器。",
                        },
                        {
                            "child_entity_id": "entity_costume",
                            "relation": "服装设定",
                            "description": "A人物的服装设计。",
                        },
                    ],
                }
            ]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://deepseek.example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.generate_entity_structure_operations(
            workspace_tree="项目/\n└── A人物/",
            current_graph={
                "nodes": [
                    {"entity_id": "entity_a", "name": "A人物", "semantic": "角色A"}
                ],
                "edges": [],
            },
            incoming_entities=[
                {
                    "entity_id": "entity_weapon",
                    "name": "A人物武器",
                    "semantic": "角色A使用的武器",
                },
                {
                    "entity_id": "entity_costume",
                    "name": "A人物服装设定",
                    "semantic": "角色A的服装设计",
                },
            ],
        )
    finally:
        await client.close()

    assert isinstance(result.operations[0], GroupEntityOperation)
    assert captured["thinking"] == {"type": "disabled"}
    messages = captured["messages"]
    assert isinstance(messages, list)
    sent = json.loads(messages[1]["content"])
    assert sent["workspace_tree"].startswith("项目/")
    assert sent["current_graph"]["nodes"][0]["entity_id"] == "entity_a"
    assert [item["entity_id"] for item in sent["incoming_entities"]] == [
        "entity_weapon",
        "entity_costume",
    ]
    assert "reason" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_entity_structure_client_rejects_omitted_incoming_entity() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        content = {"operations": [{"type": "separate", "entity_ids": ["entity_a"]}]}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://deepseek.example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(DoubaoResponseError, match="omitted"):
            await client.generate_entity_structure_operations(
                workspace_tree="项目/",
                current_graph={"nodes": [], "edges": []},
                incoming_entities=[
                    {"entity_id": "entity_a", "name": "A", "semantic": "角色A"},
                    {"entity_id": "entity_b", "name": "B", "semantic": "角色B"},
                ],
            )
    finally:
        await client.close()
