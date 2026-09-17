"""Synchronous Agent-memory recall through the shared text-embedding/RAG route."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Protocol

from capsule.agent.contracts import MemoryWrite
from capsule.db.agent_memory import AgentConversationRepository, PostgresAgentMemoryStore
from capsule.schemas import EmbeddingResult
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore, AgentMemoryVectorHit

logger = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    async def embed_text(self, text: str) -> EmbeddingResult: ...


class MilvusAgentMemoryStore:
    """Recall memory IDs from Milvus, then authorize and rank in PostgreSQL.

    A vector is never treated as authority: scope filtering happens in both
    Milvus and PostgreSQL, and a stale vector version is rejected during
    PostgreSQL hydration.  Failed vector infrastructure degrades to the
    existing lexical reader so a recall outage cannot fail an Agent turn.
    """

    def __init__(
        self,
        *,
        repository: AgentConversationRepository,
        embedder: TextEmbedder,
        vector_store: AgentMemoryMilvusStore,
        per_scope_limit: int,
        candidate_multiplier: int,
    ) -> None:
        if per_scope_limit < 1:
            raise ValueError("per_scope_limit must be positive")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be positive")
        self._repository = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._per_scope_limit = per_scope_limit
        self._candidate_multiplier = candidate_multiplier
        self._fallback = PostgresAgentMemoryStore(
            repository,
            per_scope_limit=per_scope_limit,
        )

    async def load(
        self,
        *,
        user_id: str,
        workspace_id: str,
        query: str,
    ) -> list[dict[str, object]]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        try:
            embedded = await self._embedder.embed_text(normalized_query)
            candidate_limit = self._per_scope_limit * self._candidate_multiplier
            workspace_hits, global_hits = await asyncio.gather(
                self._vector_store.search(
                    vector=embedded.vector,
                    scope="workspace",
                    user_id=user_id,
                    workspace_id=workspace_id,
                    limit=candidate_limit,
                ),
                self._vector_store.search(
                    vector=embedded.vector,
                    scope="global",
                    user_id=user_id,
                    workspace_id=workspace_id,
                    limit=candidate_limit,
                ),
            )
            return await self._repository.retrieve_memory_context_from_vectors(
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_candidates=_vector_candidates(workspace_hits),
                global_candidates=_vector_candidates(global_hits),
                per_scope_limit=self._per_scope_limit,
            )
        except Exception:
            logger.warning(
                "agent memory vector recall failed; falling back to lexical recall",
                exc_info=True,
            )
            return await self._fallback.load(
                user_id=user_id,
                workspace_id=workspace_id,
                query=normalized_query,
            )

    async def save(
        self,
        *,
        user_id: str,
        workspace_id: str,
        writes: Iterable[MemoryWrite],
    ) -> None:
        await self._fallback.save(
            user_id=user_id,
            workspace_id=workspace_id,
            writes=writes,
        )


def _vector_candidates(hits: list[AgentMemoryVectorHit]) -> list[tuple[str, float, int]]:
    """Keep repository contracts storage-neutral while preserving vector versions."""

    return [(item.memory_id, item.distance, item.memory_version) for item in hits]
