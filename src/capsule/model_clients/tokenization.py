"""Local and Ark-backed token counters with task-local content caches."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx
from tokenizers import Tokenizer

from capsule.config import Settings
from capsule.model_clients.doubao import DoubaoConfigurationError, DoubaoResponseError


class TokenCounter(Protocol):
    async def count_many(self, texts: Sequence[str]) -> list[int]: ...


DEFAULT_LOCAL_TOKENIZER_PATH = (
    Path(__file__).parent / "tokenizers" / "deepseek_v3" / "tokenizer.json"
)
LOCAL_TOKENIZER_ID = "deepseek-v3-bpe:ecb6f9fc36989434"


class LocalTokenCounter:
    """Count raw-text tokens locally with the bundled DeepSeek V3 tokenizer."""

    def __init__(self, tokenizer_path: Path | None = None, *, batch_size: int = 64) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.tokenizer_path = (tokenizer_path or DEFAULT_LOCAL_TOKENIZER_PATH).expanduser()
        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(f"local tokenizer file does not exist: {self.tokenizer_path}")
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._batch_size = batch_size
        self._cache: dict[str, int] = {}

    async def count_many(self, texts: Sequence[str]) -> list[int]:
        keys = [_cache_key(LOCAL_TOKENIZER_ID, text) for text in texts]
        missing: dict[str, str] = {}
        for key, value in zip(keys, texts, strict=True):
            if key not in self._cache:
                missing[key] = value

        missing_items = list(missing.items())
        for start in range(0, len(missing_items), self._batch_size):
            batch = missing_items[start : start + self._batch_size]
            encodings = self._tokenizer.encode_batch(
                [text for _, text in batch],
                add_special_tokens=False,
            )
            for (key, _), encoding in zip(batch, encodings, strict=True):
                self._cache[key] = len(encoding.ids)
        return [self._cache[key] for key in keys]


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
