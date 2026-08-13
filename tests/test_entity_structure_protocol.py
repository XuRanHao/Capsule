import json

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
)


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
