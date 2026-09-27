import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from capsule.agent.memory import reciprocal_rank_fuse_memory_context
from capsule.agent.memory_intent import MemoryRecallIntent, ModelMemoryIntentRecognizer


def test_rrf_fuses_multiple_memory_queries_without_a_second_rerank() -> None:
    fused = reciprocal_rank_fuse_memory_context(
        [
            [
                {"memory_id": "export", "text": "导出为 Markdown"},
                {"memory_id": "language", "text": "使用中文"},
            ],
            [
                {"memory_id": "language", "text": "使用中文"},
                {"memory_id": "archive", "text": "保留归档"},
            ],
        ]
    )

    assert [item["memory_id"] for item in fused] == ["language", "export"]


class RecordingIntentModel:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def generate_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        output_type: type[MemoryRecallIntent],
        schema_name: str,
        max_output_tokens: int,
        model: str | None = None,
    ) -> MemoryRecallIntent:
        self.requests.append(
            {
                "messages": list(messages),
                "output_type": output_type,
                "schema_name": schema_name,
                "max_output_tokens": max_output_tokens,
                "model": model,
            }
        )
        return MemoryRecallIntent(
            extra_queries=["项目导出格式", "用户语言偏好"]
        )


@pytest.mark.asyncio
async def test_model_intent_recognizer_uses_only_short_term_context() -> None:
    model = RecordingIntentModel()
    recognizer = ModelMemoryIntentRecognizer(
        model=model,
        model_name="intent-test-model",
        max_output_tokens=256,
    )

    queries = await recognizer.recognize(
        user_input="按上次格式导出",
        short_term_context={
            "summary": "讨论过导出规范。",
            "recent_messages": [{"role": "assistant", "content": "已确认 Markdown。"}],
        },
    )

    assert queries == ["项目导出格式", "用户语言偏好"]
    request = model.requests[0]
    assert request["output_type"] is MemoryRecallIntent
    assert request["schema_name"] == "capsule_agent_memory_recall_intent"
    messages = request["messages"]
    assert isinstance(messages, list)
    payload_message = messages[1]
    assert isinstance(payload_message, Mapping)
    payload = json.loads(str(payload_message["content"]))
    assert payload["original_input"] == "按上次格式导出"
    assert "tool_history" not in payload["short_term_context"]
