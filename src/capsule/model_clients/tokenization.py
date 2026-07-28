"""Batched Ark token counting with a task-local content cache."""

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoConfigurationError, DoubaoResponseError


class TokenCounter(Protocol):
    async def count_many(self, texts: Sequence[str]) -> list[int]: ...


class ArkTokenCounter:
    def __init__(self, settings: Settings) -> None:
        if settings.ark_api_key is None:
            raise DoubaoConfigurationError("CAPSULE_ARK_API_KEY is required for tokenization")
        self._settings = settings
        self._cache: dict[str, int] = {}
        self._client = httpx.AsyncClient(
            base_url=settings.ark_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.ark_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ArkTokenCounter":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def count_many(self, texts: Sequence[str]) -> list[int]:
        keys = [_cache_key(self._settings.embedding_model, text) for text in texts]
        missing: dict[str, str] = {}
        for key, value in zip(keys, texts, strict=True):
            if key not in self._cache:
                missing[key] = value

        missing_items = list(missing.items())
        batch_size = self._settings.tokenization_batch_size
        for start in range(0, len(missing_items), batch_size):
            batch = missing_items[start : start + batch_size]
            counts = await self._request([text for _, text in batch])
            for (key, _), count in zip(batch, counts, strict=True):
                self._cache[key] = count
        return [self._cache[key] for key in keys]

    async def _request(self, texts: list[str]) -> list[int]:
        response = await self._client.post(
            "/tokenization",
            json={"model": self._settings.embedding_model, "text": texts},
            timeout=self._settings.embedding_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise DoubaoResponseError("tokenization response length does not match request")
        ordered: list[int | None] = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                raise DoubaoResponseError("tokenization item must be an object")
            index = item.get("index")
            total = item.get("total_tokens")
            if not isinstance(index, int) or not 0 <= index < len(texts):
                raise DoubaoResponseError("tokenization item has an invalid index")
            if not isinstance(total, int) or total < 0:
                raise DoubaoResponseError("tokenization item has an invalid token count")
            ordered[index] = total
        if any(value is None for value in ordered):
            raise DoubaoResponseError("tokenization response is missing an item")
        return [value for value in ordered if value is not None]


def _cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"


def token_counts_from_payload(payload: dict[str, Any]) -> list[int]:
    """Reserved parsing seam for recorded API fixtures."""
    data = payload.get("data")
    if not isinstance(data, list):
        raise DoubaoResponseError("tokenization response data must be a list")
    return [int(item["total_tokens"]) for item in sorted(data, key=lambda item: item["index"])]
