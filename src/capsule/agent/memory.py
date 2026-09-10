"""Long-term memory boundary; short-term conversation lives in AgentState."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Protocol

from capsule.agent.contracts import MemoryWrite


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
