import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from capsule.config import Settings
from capsule.enums import AssetType, EmbeddingType
from capsule.features import ACTIVE_EMBEDDING_TYPES, feature_dimension_scope_prompt
from capsule.model_clients.doubao import (
    DoubaoClient,
    DoubaoResponseError,
    _extract_embedding,
    _responses_content,
)


def test_responses_content_converts_direct_audio_input() -> None:
    assert _responses_content(
        [{"type": "audio_url", "audio_url": "data:audio/wav;base64,ZmFrZQ=="}]
    ) == [
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZmFrZQ=="}
    ]


@pytest.mark.asyncio
async def test_asset_entity_relations_use_parallel_batches_and_preserve_order() -> None:
    batch_sizes: list[int] = []
    active = 0
    max_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        payload = json.loads(request.content)
        candidates = json.loads(payload["messages"][1]["content"])["candidates"]
        batch_sizes.append(len(candidates))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        relations = [
            {
                "source_id": item["source_id"],
                "target_id": item["target_id"],
                "establishes_relation": True,
                "relation": "相关",
                "description": "测试关系",
            }
            for item in candidates
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"relations": relations})}}]},
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
    candidates = [{"source_id": f"asset_{index}", "target_id": "entity_1"} for index in range(13)]
    try:
        resolution = await client.generate_asset_entity_relations(candidates)
    finally:
        await client.close()

    assert sorted(batch_sizes) == [3, 10]
    assert max_active == 2
    assert [item.source_id for item in resolution.relations] == [
        item["source_id"] for item in candidates
    ]


def test_extract_embedding_accepts_openai_list_shape() -> None:
    assert _extract_embedding({"data": [{"embedding": [[1, 2.5]]}]}) == [1.0, 2.5]


def test_extract_embedding_accepts_ark_object_shape() -> None:
    assert _extract_embedding({"data": {"embedding": [[1, 2.5]]}}) == [1.0, 2.5]


def test_extract_embedding_rejects_missing_vector() -> None:
    with pytest.raises(DoubaoResponseError, match="does not contain a vector"):
        _extract_embedding({"data": {}})


@pytest.mark.asyncio
async def test_dimension_selector_sends_all_dimensions_and_resolves_weights() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "embedding_types": [
                                        "native_multimodal",
                                        "visual_presentation",
                                    ],
                                    "weights": {
                                        "native_multimodal": 0.3,
                                        "visual_presentation": 0.7,
                                    },
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
            search_query_max_output_tokens=500,
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://deepseek.example.test",
        headers={"Authorization": "Bearer deepseek-test-key"},
        transport=httpx.MockTransport(handler),
    )
    try:
        suggestion = await client.select_search_dimensions(
            query_text="想找蓝紫色占满画面、明暗反差很强的动画场景，人物是谁无所谓",
            asset_types=[AssetType.IMAGE, AssetType.VIDEO_SEGMENT],
        )
    finally:
        await client.close()

    assert suggestion.embedding_types == [
        EmbeddingType.NATIVE_MULTIMODAL,
        EmbeddingType.VISUAL_PRESENTATION,
    ]
    assert suggestion.weights == {
        EmbeddingType.NATIVE_MULTIMODAL: 0.3,
        EmbeddingType.VISUAL_PRESENTATION: 0.7,
    }
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["max_tokens"] == 256
    messages = captured["messages"]
    assert isinstance(messages, list)
    payload = json.loads(messages[1]["content"])
    assert payload["query_text"] == ("想找蓝紫色占满画面、明暗反差很强的动画场景，人物是谁无所谓")
    assert payload["target_asset_types"] == ["image", "video_segment"]
    assert len(payload["candidate_dimensions"]) == len(ACTIVE_EMBEDDING_TYPES)
    assert {item["embedding_type"] for item in payload["candidate_dimensions"]} == {
        item.value for item in ACTIVE_EMBEDDING_TYPES
    }


@pytest.mark.asyncio
async def test_asset_and_search_understanding_use_independent_pools() -> None:
    feature_names = [
        "subject_content",
        "scene_theme",
        "visual_presentation",
    ]

    async def ark_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/responses"
        content = {
            "asset_name": "测试素材",
            "asset_description": "一条用于验证独立并发池的素材描述。",
            "features": {
                name: {
                    "value": "测试值",
                    "status": "observed",
                    "confidence": 0.9,
                    "evidence": ["测试证据"],
                }
                for name in feature_names
            },
        }
        return httpx.Response(200, json={"output_text": json.dumps(content)})

    async def deepseek_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        content = {
            "queries": {
                "native_multimodal": "重点看视觉风格，内容其次",
                "visual_presentation": "视觉风格",
            },
            "weights": {
                "native_multimodal": 0.25,
                "visual_presentation": 0.75,
            },
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
        )

    client = DoubaoClient(
        Settings(
            ark_api_key=SecretStr("test-key"),
            deepseek_api_key=SecretStr("deepseek-test-key"),
            understanding_concurrency=1,
            search_understanding_concurrency=1,
        )
    )
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(ark_handler),
    )
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://deepseek.example.test",
        transport=httpx.MockTransport(deepseek_handler),
    )
    try:
        await client.understand_asset([{"role": "user", "content": "分析测试素材"}])
        await client.enhance_search_query(
            query_text="重点看视觉风格，内容其次",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.VISUAL_PRESENTATION,
            ],
        )
    finally:
        await client.close()

    assert client.asset_understanding_pool.max_observed == 1
    assert client.search_understanding_pool.max_observed == 1


@pytest.mark.asyncio
async def test_query_enhancer_uses_deepseek_chat_with_bounded_non_thinking_output() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer deepseek-test-key"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "queries": {
                                        "native_multimodal": "蓝色湖泊",
                                        "visual_presentation": "清透的自然摄影风格",
                                    },
                                    "weights": {
                                        "native_multimodal": 0.2,
                                        "visual_presentation": 0.6,
                                    },
                                },
                                ensure_ascii=False,
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
            search_query_max_output_tokens=400,
        )
    )
    await client.close()
    client._deepseek_client = httpx.AsyncClient(
        base_url="https://example.test",
        headers={"Authorization": "Bearer deepseek-test-key"},
        transport=httpx.MockTransport(handler),
    )
    try:
        enhancement = await client.enhance_search_query(
            query_text="蓝色湖泊，重点看清透自然摄影风格，原始内容其次",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.VISUAL_PRESENTATION,
            ],
        )
    finally:
        await client.close()

    assert captured["thinking"] == {"type": "disabled"}
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["max_tokens"] == 400
    assert captured["response_format"] == {"type": "json_object"}
    assert "根节点必须且只能包含 queries 和 weights" in str(captured["messages"])
    assert '"required_embedding_types": ["native_multimodal", "visual_presentation"]' in str(
        captured["messages"]
    )
    assert "原始内容" in str(captured["messages"])
    assert "摄影、插画、三维渲染等媒介与成像方式" in str(captured["messages"])
    assert "不能把其中的类别示例或枚举词复制进 query" in str(captured["messages"])
    assert "以目标维度为中心提高该维度信息密度" in str(captured["messages"])
    assert "只保留必要上下文" in str(captured["messages"])
    assert "允许保留能说明目标维度的跨维度关联" in str(captured["messages"])
    assert "不要机械地按词或维度删除" in str(captured["messages"])
    assert "保守的维度化表达" in str(captured["messages"])
    assert "权重控制意图应体现在 weights 中" in str(captured["messages"])
    assert '"scene_theme"' not in str(captured["messages"])
    assert "蓝色湖泊，重点看清透自然摄影风格，原始内容其次" in str(captured["messages"])
    assert "input_image" not in str(captured["messages"])
    assert enhancement.queries == {
        EmbeddingType.NATIVE_MULTIMODAL: "蓝色湖泊",
        EmbeddingType.VISUAL_PRESENTATION: "清透的自然摄影风格",
    }
    assert enhancement.weights == pytest.approx(
        {
            EmbeddingType.NATIVE_MULTIMODAL: 0.25,
            EmbeddingType.VISUAL_PRESENTATION: 0.75,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": 0.5, "visual_presentation": 0.5},
            "source": "不应出现",
        },
        {
            "queries": {"native_multimodal": "原查询", "scene_theme": "场景"},
            "weights": {"native_multimodal": 0.5, "visual_presentation": 0.5},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": 0.5, "scene_theme": 0.5},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "   "},
            "weights": {"native_multimodal": 0.5, "visual_presentation": 0.5},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": 0.0, "visual_presentation": 1.0},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": float("inf"), "visual_presentation": 1.0},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": True, "visual_presentation": 1.0},
        },
        {
            "queries": {"native_multimodal": "原查询", "visual_presentation": "风格"},
            "weights": {"native_multimodal": "0.5", "visual_presentation": 0.5},
        },
    ],
)
async def test_query_enhancer_rejects_invalid_output(content: object) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
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
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(DoubaoResponseError):
            await client.enhance_search_query(
                query_text="重点看视觉风格",
                embedding_types=[
                    EmbeddingType.NATIVE_MULTIMODAL,
                    EmbeddingType.VISUAL_PRESENTATION,
                ],
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_understand_asset_constrains_object_schema_and_repairs_invalid_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    calls: list[dict[str, object]] = []
    feature_names = [
        "subject_content",
        "scene_theme",
        "visual_presentation",
    ]
    valid_features = {
        name: {
            "applicability": "applicable",
            "items": [
                {
                    "description": "测试值",
                    "salience": "high",
                    "status": "observed",
                    "evidence": ["测试证据"],
                    "ocr_confidence": None,
                }
            ],
        }
        for name in feature_names
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/responses"
        payload = json.loads(request.content)
        calls.append(payload)
        features: object = (
            [{**valid_features["subject_content"]}] if len(calls) == 1 else valid_features
        )
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "asset_name": "测试素材",
                        "asset_description": "一条用于验证结构化输出约束的素材描述。",
                        "features": features,
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
        result = await client.understand_asset(
            [
                {
                    "role": "system",
                    "content": (
                        f"三个 Feature 的正向语义范围如下：{feature_dimension_scope_prompt()}。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "分析这条测试素材"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,ZmFrZQ=="},
                        },
                    ],
                },
            ],
            asset_id="asset_structure_repair",
        )
    finally:
        await client.close()

    assert result.features.subject_content.value == "测试值"
    assert len(calls) == 2
    assert calls[0]["thinking"] == {"type": "disabled"}
    assert calls[0]["max_output_tokens"] == 2048
    response_format = calls[0]["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "asset_understanding"
    assert response_format["strict"] is True
    assert response_format["schema"]["additionalProperties"] is False
    assert response_format["schema"]["required"] == [
        "asset_name",
        "asset_description",
        "features",
        "transcript",
    ]
    feature_item_schema = response_format["schema"]["$defs"]["FeatureItem"]
    assert "description" in feature_item_schema["properties"]
    assert "description" in feature_item_schema["required"]
    assert (
        feature_item_schema["properties"]["description"]["description"]
        == "当前维度内的一条事实本身，不是证据说明"
    )
    subject_item_schema = response_format["schema"]["$defs"]["SubjectFeatureItem"]
    assert "subject" in subject_item_schema["properties"]
    assert "subject" in subject_item_schema["required"]
    assert "可跨素材复用的主体名称" in (subject_item_schema["properties"]["subject"]["description"])
    first_input = calls[0]["input"]
    assert isinstance(first_input, list)
    assert "features 必须是对象且包含 Schema 指定的三个字段" in str(first_input[0])
    assert "三个 Feature 的正向语义范围" not in str(first_input[0])
    assert str(first_input).count("三个 Feature 的正向语义范围") == 1
    assert "scene_theme=聚焦整幅素材所呈现的全局情境" in str(first_input)
    assert "applicability=not_applicable" in str(first_input)
    assert "visual_presentation=聚焦素材如何被视觉化呈现" in str(first_input)
    assert "description 表示当前维度的事实本身" in str(first_input[0])
    assert "subject 是主体名称，description 是主体描述" in str(first_input[0])
    assert "不能用 evidence 代替description" in str(first_input[0])
    assert "主辅色、冷暖、饱和度、明度、对比" in str(first_input)
    assert "跨主体、跨场景找到呈现方式相近" in str(first_input)
    assert "JSON 结构示例" not in str(first_input[0])
    assert all(name in str(response_format["schema"]) for name in feature_names)
    assert "input_image" in str(first_input)
    assert "上一份输出未通过 AssetUnderstanding 结构校验" in str(calls[1]["input"])
    repair_input = str(calls[1]["input"])
    assert "十个命名字段" not in repair_input
    assert "Schema 指定的 3 个命名字段" in repair_input
    assert all(name in repair_input for name in feature_names)
    assert "asset_id=asset_structure_repair" in caplog.text
    assert "error_type=ValidationError" in caplog.text
    assert "first_attempt_ms=" in caplog.text
    assert "repair_method=model_retry" in caplog.text
    assert "repair_ms=" in caplog.text


@pytest.mark.asyncio
async def test_understand_asset_normalizes_unambiguous_shape_without_model_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "asset_name": "测试素材",
                        "asset_description": "一条用于验证本地结构归一化的素材描述。",
                        "features": [
                            {
                                "key": "subject_content",
                                "value": ["圆形角色", "挥手动作"],
                                "status": "observed",
                                "confidence": "0.8",
                                "evidence": "画面中可见角色挥手",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            },
        )

    caplog.set_level("INFO")
    client = DoubaoClient(Settings(ark_api_key=SecretStr("test-key")))
    await client.close()
    client._client = httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.understand_asset(
            [{"role": "user", "content": "分析测试素材"}],
            asset_id="asset_local_normalization",
        )
    finally:
        await client.close()

    assert len(calls) == 1
    assert result.features.subject_content.value == "圆形角色；挥手动作"
    assert result.features.subject_content.items[0].evidence == ["画面中可见角色挥手"]
    assert result.features.scene_theme.applicability.value == "unknown"
    assert "asset_id=asset_local_normalization" in caplog.text
    assert "repair_method=local_normalization" in caplog.text
    assert "request_attempt=first" in caplog.text
    assert "request_ms=" in caplog.text
    assert "feature_array_to_object" in caplog.text
    assert "legacy_feature_shape_expanded" in caplog.text


@pytest.mark.asyncio
async def test_understand_asset_attempts_at_most_one_model_repair() -> None:
    calls: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output_text": json.dumps(
                    {
                        "asset_name": "测试素材",
                        "asset_description": "一条持续返回歧义结构的测试素材描述。",
                        "features": [{"value": "无法确定所属维度"}],
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
        with pytest.raises(ValidationError, match="features must be an object"):
            await client.understand_asset(
                [{"role": "user", "content": "分析测试素材"}],
                asset_id="asset_repair_limit",
            )
    finally:
        await client.close()

    assert len(calls) == 2
    assert "上一份输出未通过 AssetUnderstanding 结构校验" in str(calls[1]["input"])


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
async def test_summarize_cluster_uses_text_model_with_thinking_disabled() -> None:
    captured: dict[str, object] = {}
    description = "共同呈现蓝紫色冷光、低饱和度和高明暗对比，整体色彩关系保持稳定。"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "name": "蓝紫色霓虹夜景",
                                    "description": description,
                                    "common_features": ["蓝紫色冷光", "城市夜景"],
                                    "internal_variance": "low",
                                },
                                ensure_ascii=False,
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
        summary = await client.summarize_cluster(
            [
                {"role": "system", "content": "只输出 JSON"},
                {"role": "user", "content": "簇内全部资产文本只有 asset_1"},
            ]
        )
    finally:
        await client.close()

    assert captured["model"] == "deepseek-v4-flash"
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["response_format"] == {"type": "json_object"}
    assert "asset_1" in str(captured["messages"])
    assert summary.name == "蓝紫色霓虹夜景"


@pytest.mark.asyncio
async def test_summarize_cluster_retries_once_when_response_violates_contract() -> None:
    calls: list[dict[str, object]] = []
    valid_description = "共同呈现蓝紫色冷光、低饱和度和高明暗对比，整体色彩关系保持稳定。"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        description = "这段描述太短。" if len(calls) == 1 else valid_description
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "name": "蓝紫色霓虹夜景",
                                    "description": description,
                                    "common_features": ["蓝紫色光线"],
                                    "internal_variance": "low",
                                },
                                ensure_ascii=False,
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
        summary = await client.summarize_cluster(
            [
                {"role": "system", "content": "只输出 JSON"},
                {"role": "user", "content": "簇内全部资产文本只有 asset_1"},
            ]
        )
    finally:
        await client.close()

    assert summary.description == valid_description
    assert len(calls) == 2
    assert "上一份输出未通过结构校验" in str(calls[1]["messages"])
    assert "description 必须是 30 到 80 个中文字符" in str(calls[1]["messages"])
    assert "common_features 必须有 1 到 3 项" in str(calls[1]["messages"])
    assert "不要输出 keywords" in str(calls[1]["messages"])
