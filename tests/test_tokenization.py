from pathlib import Path

import httpx
import pytest

from capsule.config import Settings
from capsule.model_clients.tokenization import ArkTokenCounter, LocalTokenCounter


@pytest.mark.asyncio
async def test_local_token_counter_uses_bundled_deepseek_v3_without_special_tokens() -> None:
    counter = LocalTokenCounter(batch_size=2)

    counts = await counter.count_many(
        [
            "这是一段用于比较本地与远程分词结果的中文文本。",
            "Hello, this is a tokenizer comparison for English text.",
            "# 标题\n\n中英文 mixed content，包含 `code()`、URL https://example.com 和数字 2026。",
        ]
    )

    assert counts == [13, 12, 26]


def test_local_token_counter_rejects_missing_override(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="local tokenizer file does not exist"):
        LocalTokenCounter(tmp_path / "missing-tokenizer.json")


@pytest.mark.asyncio
async def test_token_counter_batches_caches_and_restores_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(variable, raising=False)
    requests: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = __import__("json").loads(request.content)
        texts = payload["text"]
        requests.append(texts)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "total_tokens": len(text)}
                    for index, text in reversed(list(enumerate(texts)))
                ]
            },
        )

    settings = Settings(
        ark_api_key="test-key",
        tokenization_batch_size=2,
    )
    counter = ArkTokenCounter(settings)
    await counter._client.aclose()
    counter._client = httpx.AsyncClient(
        base_url=settings.ark_base_url,
        transport=httpx.MockTransport(handler),
    )
    try:
        first = await counter.count_many(["a", "bb", "ccc", "a"])
        second = await counter.count_many(["ccc", "a"])
    finally:
        await counter.close()

    assert first == [1, 2, 3, 1]
    assert second == [3, 1]
    assert requests == [["a", "bb"], ["ccc"]]
