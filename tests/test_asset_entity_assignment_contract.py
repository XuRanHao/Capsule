import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoClient, DoubaoResponseError
from capsule.relation_graph import (
    AssetEntityAssignmentRequest,
    AssetEntityAssignmentResolution,
)


def _assignment_request() -> AssetEntityAssignmentRequest:
    return AssetEntityAssignmentRequest.model_validate(
        {
            "assignments": [
                {
                    "asset": {
                        "asset_id": "asset_1",
                        "metadata": {"source_path": "characters/a/weapon.png"},
                        "content_description": "A character weapon concept image",
                        "content_subject": "A's sword",
                    },
                    "candidates": [
                        {
                            "entity_id": "entity_weapon",
                            "name": "A weapon",
                            "semantic": "Weapon used by character A",
                        },
                        {
                            "entity_id": "entity_costume",
                            "name": "A costume",
                            "semantic": "Costume design for character A",
                        },
                    ],
                }
            ]
        }
    )


def test_assignment_contract_requires_exact_coverage_and_topk_membership() -> None:
    request = _assignment_request()
    request.validate_resolution(
        AssetEntityAssignmentResolution.model_validate(
            {
                "assignments": [
                    {
                        "asset_id": "asset_1",
                        "entity_id": "entity_weapon",
                        "reason": "The asset content and path both identify A's sword.",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="one of the Asset candidates"):
        request.validate_resolution(
            AssetEntityAssignmentResolution.model_validate(
                {
                    "assignments": [
                        {
                            "asset_id": "asset_1",
                            "entity_id": "entity_outside_top_k",
                            "reason": "Not a permitted candidate.",
                        }
                    ]
                }
            )
        )

    with pytest.raises(ValueError, match="cover each input asset exactly once"):
        request.validate_resolution(AssetEntityAssignmentResolution())


@pytest.mark.asyncio
async def test_assignment_client_batches_and_sends_topk_constrained_prompt() -> None:
    batch_sizes: list[int] = []
    prompt_texts: list[str] = []
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        payload = json.loads(request.content)
        prompt_texts.append(payload["messages"][0]["content"])
        batch = json.loads(payload["messages"][1]["content"])["assignments"]
        batch_sizes.append(len(batch))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        result = {
            "assignments": [
                {
                    "asset_id": item["asset"]["asset_id"],
                    "entity_id": (
                        None
                        if item["asset"]["asset_id"] == "asset_10"
                        else item["candidates"][0]["entity_id"]
                    ),
                    "reason": (
                        "No reliable candidate."
                        if item["asset"]["asset_id"] == "asset_10"
                        else "Direct semantic match."
                    ),
                }
                for item in batch
            ]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(result)}}]},
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    assignments = [
        {
            "asset": {
                "asset_id": f"asset_{index}",
                "metadata": {"source_path": f"characters/a/item_{index}.png"},
                "content_description": "Character A concept material",
                "content_subject": "Character A item",
            },
            "candidates": [
                {
                    "entity_id": "entity_a_items",
                    "name": "A items",
                    "semantic": "Items belonging to character A",
                }
            ],
        }
        for index in range(11)
    ]
    try:
        resolution = await client.assign_assets_to_entities(assignments)
    finally:
        await client.close()

    assert sorted(batch_sizes) == [1, 10]
    assert max_active == 2
    assert "entity_id 只能取" in prompt_texts[0]
    assert [assignment.asset_id for assignment in resolution.assignments] == [
        item["asset"]["asset_id"] for item in assignments
    ]
    assert resolution.assignments[10].entity_id is None
    assert resolution.assignments[10].reason == "No reliable candidate."
    assert resolution.assignments[10].description == ""


@pytest.mark.asyncio
async def test_assignment_client_rejects_entity_outside_the_asset_topk() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assignments": [
                                        {
                                            "asset_id": "asset_1",
                                            "entity_id": "entity_outside_top_k",
                                            "reason": "Incorrect candidate.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(DoubaoResponseError, match="must be one of the Asset candidates"):
            await client.assign_assets_to_entities(
                _assignment_request().model_dump()["assignments"]
            )
    finally:
        await client.close()
