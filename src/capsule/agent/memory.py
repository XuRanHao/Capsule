"""Long-term memory boundary; short-term conversation lives in AgentState."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from capsule.agent.contracts import MemoryWrite


def reciprocal_rank_fuse_memory_context(
    result_lists: Sequence[Sequence[Mapping[str, Any]]],
    *,
    rank_constant: int = 60,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse per-query memory rankings while preserving the existing context shape."""

    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    effective_limit = (
        limit
        if limit is not None
        else max((len(items) for items in result_lists), default=0)
    )
    if effective_limit < 1:
        return []

    scores: dict[str, float] = {}
    values: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    next_seen = 0
    for items in result_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(items, start=1):
            record = dict(item)
            identity = _memory_context_identity(record)
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            if identity not in values:
                values[identity] = record
                first_seen[identity] = next_seen
                next_seen += 1
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rank_constant + rank)

    ranked = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [values[key] for key in ranked[:effective_limit]]


def _memory_context_identity(record: Mapping[str, Any]) -> str:
    memory_id = record.get("memory_id")
    if isinstance(memory_id, str) and memory_id:
        return f"memory:{memory_id}"
    scope = record.get("scope")
    key = record.get("key")
    if isinstance(scope, str) and isinstance(key, str) and scope and key:
        return f"scope-key:{scope}:{key}"
    return f"value:{json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str)}"


class AgentMemoryStore(Protocol):
    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, Any]]: ...

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None: ...


class NullMemoryStore:
    """Default store used until the project-specific memory repository is wired in."""

    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return []

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None:
        return None


class DelegatingMemoryStore:
    """Stable graph dependency whose concrete memory reader can change after startup."""

    def __init__(self, delegate: AgentMemoryStore) -> None:
        self._delegate = delegate

    def set_delegate(self, delegate: AgentMemoryStore) -> None:
        self._delegate = delegate

    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, Any]]:
        return await self._delegate.load(
            user_id=user_id,
            workspace_id=workspace_id,
            query=query,
        )

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None:
        await self._delegate.save(
            user_id=user_id,
            workspace_id=workspace_id,
            writes=writes,
        )


class InMemoryMemoryStore:
    """Small deterministic implementation for local development and unit tests."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, Any]]:
        del query
        async with self._lock:
            values = self._values[(user_id, workspace_id)]
            return [{"key": key, "value": value} for key, value in values.items()]

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None:
        async with self._lock:
            values = self._values[(user_id, workspace_id)]
            for write in writes:
                values[write.key] = write.value
