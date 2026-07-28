import httpx
import pytest

from capsule.config import Settings
from capsule.model_clients.tokenization import ArkTokenCounter


@pytest.mark.asyncio
async def test_token_counter_batches_caches_and_restores_response_order() -> None:
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
