import json

import httpx
import pytest
from pydantic import SecretStr

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoClient


def _client(handler: httpx.MockTransport) -> DoubaoClient:
    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
        )
    )
    return client


@pytest.mark.asyncio
async def test_client_selects_related_top_level_tree_pair() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        content = {
            "pairs": [
                {
                    "source_entity_id": "entity_character",
                    "target_entity_id": "entity_hospital",
                }
            ]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = _client(httpx.MockTransport(handler))
    await client.close()
    client._deepseek_client = httpx.AsyncClient(  # noqa: SLF001
        base_url="https://deepseek.example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        resolution = await client.select_related_entity_pairs(
            workspace_tree="项目/\n├── 角色/\n└── 医院/",
            current_entities=[
                {
                    "entity_id": "entity_character",
                    "name": "人物A",
                    "semantic": "在医院剧情出现的人物",
                }
            ],
            incoming_entities=[
                {
                    "entity_id": "entity_hospital",
                    "name": "医院",
                    "semantic": "故事中的医院场景",
                }
            ],
        )
    finally:
        await client.close()

    assert len(resolution.pairs) == 1
    assert "只做候选召回" in captured["messages"][0]["content"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_client_connects_non_root_nodes_across_two_trees() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        content = {
            "relations": [
                {
                    "source_entity_id": "entity_character_normal",
                    "target_entity_id": "entity_bed",
                    "relation": "躺在",
                    "description": "人物正常形态出现在病床上。",
                }
            ]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = _client(httpx.MockTransport(handler))
    await client.close()
    client._deepseek_client = httpx.AsyncClient(  # noqa: SLF001
        base_url="https://deepseek.example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        resolution = await client.generate_cross_tree_relations(
            workspace_tree="项目/",
            left_tree={
                "root_entity_id": "entity_character",
                "nodes": [
                    {
                        "entity_id": "entity_character",
                        "name": "人物A",
                        "semantic": "人物A",
                    },
                    {
                        "entity_id": "entity_character_normal",
                        "name": "人物A正常形态",
                        "semantic": "人物A的正常状态",
                    },
                ],
                "edges": [
                    {
                        "source_entity_id": "entity_character_normal",
                        "target_entity_id": "entity_character",
                        "relation": "正常形态",
                        "description": "人物A的正常形态。",
                    }
                ],
            },
            right_tree={
                "root_entity_id": "entity_hospital",
                "nodes": [
                    {
                        "entity_id": "entity_hospital",
                        "name": "医院",
                        "semantic": "医院场景",
                    },
                    {
                        "entity_id": "entity_bed",
                        "name": "病床",
                        "semantic": "医院病床",
                    },
                ],
                "edges": [
                    {
                        "source_entity_id": "entity_bed",
                        "target_entity_id": "entity_hospital",
                        "relation": "内部设施",
                        "description": "医院内的病床。",
                    }
                ],
            },
        )
    finally:
        await client.close()

    assert resolution.relations[0].source_entity_id == "entity_character_normal"
    assert resolution.relations[0].target_entity_id == "entity_bed"
