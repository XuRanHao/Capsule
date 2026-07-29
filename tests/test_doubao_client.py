import json

import httpx
import pytest
from pydantic import SecretStr

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoClient, DoubaoResponseError, _extract_embedding


def test_extract_embedding_accepts_openai_list_shape() -> None:
    assert _extract_embedding({"data": [{"embedding": [[1, 2.5]]}]}) == [1.0, 2.5]


def test_extract_embedding_accepts_ark_object_shape() -> None:
    assert _extract_embedding({"data": {"embedding": [[1, 2.5]]}}) == [1.0, 2.5]


def test_extract_embedding_rejects_missing_vector() -> None:
    with pytest.raises(DoubaoResponseError, match="does not contain a vector"):
        _extract_embedding({"data": {}})


@pytest.mark.asyncio
async def test_understand_asset_constrains_object_schema_and_repairs_invalid_shape() -> None:
    calls: list[dict[str, object]] = []
    feature_names = [
        "subject_content",
        "scene_theme",
        "visual_style",
        "color_composition",
        "mood_atmosphere",
        "character_state_or_psychology",
        "asset_usage",
        "target_audience",
        "provenance",
        "rights_version_authorship",
    ]
    valid_features = {
        name: {
            "value": "测试值",
            "status": "observed",
            "confidence": 0.9,
            "evidence": ["测试证据"],
        }
        for name in feature_names
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        features: object = (
            [{"key": "subject_content", **valid_features["subject_content"]}]
            if len(calls) == 1
            else valid_features
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "asset_name": "测试素材",
                                    "asset_description": "一条用于验证结构化输出约束的素材描述。",
                                    "features": features,
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    client = DoubaoClient(Settings(ark_api_key=SecretStr("test-key")))
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.understand_asset(
            [{"role": "user", "content": "分析这条测试素材"}]
        )
    finally:
        await client.close()

    assert result.features.subject_content.value == "测试值"
    assert len(calls) == 2
    assert calls[0]["response_format"] == {"type": "json_object"}
    first_messages = calls[0]["messages"]
    assert isinstance(first_messages, list)
    assert "features 必须是对象，不能是数组" in str(first_messages[0])
    assert "JSON 结构示例" in str(first_messages[0])
    assert "禁止照抄" in str(first_messages[0])
    assert all(name in str(first_messages[0]) for name in feature_names)
    assert "上一份输出未通过 AssetUnderstanding 结构校验" in str(calls[1]["messages"])


@pytest.mark.asyncio
async def test_embed_multimodal_requests_configured_dimension() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"data": {"embedding": [[0.0, 1.0, 2.0]]}, "model": "fake"},
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_dimension=3,
        )
    )
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.embed_text("hello")
    finally:
        await client.close()

    assert captured["dimensions"] == 3
    assert result.vector == [0.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_summarize_cluster_uses_responses_api_with_thinking_disabled() -> None:
    captured: dict[str, object] = {}
    description = (
        "这一组素材以夜间城市中的蓝紫色霓虹光影为主，人物和街道在冷色调反射中呈现出"
        "稳定的赛博朋克电影感，少量镜头的构图变化不影响整体风格判断。"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/responses"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "name": "蓝紫色霓虹夜景",
                                        "description": description,
                                        "keywords": ["霓虹", "夜景", "赛博朋克"],
                                        "common_features": ["蓝紫色冷光", "城市夜景"],
                                        "internal_variance": "low",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                    }
                ]
            },
        )

    client = DoubaoClient(Settings(ark_api_key=SecretStr("test-key")))
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        summary = await client.summarize_cluster(
            [
                {"role": "system", "content": "只输出 JSON"},
                {"role": "user", "content": "代表资产只有 asset_1"},
            ]
        )
    finally:
        await client.close()

    assert captured["model"] == "doubao-seed-2-0-lite-260215"
    assert captured["thinking"] == {"type": "disabled"}
    assert "asset_1" in str(captured["input"])
    assert summary.name == "蓝紫色霓虹夜景"


@pytest.mark.asyncio
async def test_summarize_cluster_retries_once_when_response_violates_contract() -> None:
    calls: list[dict[str, object]] = []
    valid_description = (
        "这一组素材围绕蓝紫色霓虹夜景展开，冷色光线、城市建筑和夜间反射共同形成"
        "稳定的电影化视觉风格，代表资产之间仅在画面主体与构图细节上存在轻微变化。"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        description = "这段描述太短。" if len(calls) == 1 else valid_description
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "name": "蓝紫色霓虹夜景",
                        "description": description,
                        "keywords": ["霓虹", "夜景", "冷色"],
                        "common_features": ["蓝紫色光线"],
                        "internal_variance": "low",
                    },
                    ensure_ascii=False,
                )
            },
        )

    client = DoubaoClient(Settings(ark_api_key=SecretStr("test-key")))
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        summary = await client.summarize_cluster(
            [
                {"role": "system", "content": "只输出 JSON"},
                {"role": "user", "content": "代表资产只有 asset_1"},
            ]
        )
    finally:
        await client.close()

    assert summary.description == valid_description
    assert len(calls) == 2
    assert "上一份输出未通过结构校验" in str(calls[1]["input"])
