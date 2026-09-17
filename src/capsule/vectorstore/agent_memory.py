"""Milvus storage for derived Agent-memory vectors.

PostgreSQL remains the authority for memory state, access boundaries and ranking.
This collection only maps a stable memory ID to its text embedding and lightweight
scope metadata so it can participate in the same vector-recall shape as assets.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal

from pymilvus import DataType, MilvusClient

from capsule.config import Settings
from capsule.vectorstore._milvus_shared import delete_count, field_dimension, validate_vector


@dataclass(frozen=True, slots=True)
class AgentMemoryVectorRecord:
    memory_id: str
    scope: Literal["workspace", "global"]
    user_id: str
    workspace_id: str | None
    kind: str
    memory_version: int
    vector: list[float]


@dataclass(frozen=True, slots=True)
class AgentMemoryVectorHit:
    memory_id: str
    distance: float
    memory_version: int


class AgentMemoryMilvusStore:
    """One isolated collection with the common Milvus upsert/search contract."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        token = settings.milvus_token.get_secret_value() if settings.milvus_token else ""
        self._client = client or MilvusClient(uri=settings.milvus_uri, token=token)
        self._collection = settings.agent_memory_milvus_collection
        self._dimension = settings.embedding_dimension
        self._batch_size = settings.milvus_batch_size
        self._search_ef = settings.search_hnsw_ef

    async def ensure_collection(self) -> bool:
        return await asyncio.to_thread(self._ensure_collection_sync)

    async def aupsert(self, records: list[AgentMemoryVectorRecord]) -> None:
        await asyncio.to_thread(self.upsert, records)

    async def adelete(self, memory_ids: list[str]) -> int:
        if not memory_ids:
            return 0
        return await asyncio.to_thread(self.delete, memory_ids)

    async def search(
        self,
        *,
        vector: list[float],
        scope: Literal["workspace", "global"],
        user_id: str,
        workspace_id: str,
        limit: int,
    ) -> list[AgentMemoryVectorHit]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.validate_vector(vector)
        filter_expression = _memory_filter(
            scope=scope,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            filter=filter_expression,
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {"ef": self._search_ef}},
            output_fields=["memory_id", "memory_version"],
        )
        return _parse_hits(raw)

    async def delete_workspace(self, workspace_id: str) -> int:
        expression = (
            f"scope == {json.dumps('workspace')} and "
            f"workspace_id == {json.dumps(workspace_id)}"
        )
        return await asyncio.to_thread(self._delete_expression, expression)

    def upsert(self, records: list[AgentMemoryVectorRecord]) -> None:
        for record in records:
            self.validate_vector(record.vector)
            if record.scope == "workspace" and not record.workspace_id:
                raise ValueError("workspace memory vector requires workspace_id")
            if record.scope == "global" and record.workspace_id is not None:
                raise ValueError("global memory vector must not have workspace_id")
        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            self._client.upsert(
                collection_name=self._collection,
                data=[
                    {
                        "memory_id": record.memory_id,
                        "scope": record.scope,
                        "user_id": record.user_id,
                        "workspace_id": record.workspace_id or "",
                        "kind": record.kind,
                        "memory_version": record.memory_version,
                        "vector": record.vector,
                    }
                    for record in batch
                ],
            )
        if records:
            self._client.flush(collection_name=self._collection)

    def delete(self, memory_ids: list[str]) -> int:
        encoded = ", ".join(json.dumps(item) for item in memory_ids)
        return self._delete_expression(f"memory_id in [{encoded}]")

    def validate_vector(self, vector: list[float]) -> None:
        validate_vector(vector, dimension=self._dimension)

    def _ensure_collection_sync(self) -> bool:
        if self._client.has_collection(collection_name=self._collection):
            description = self._client.describe_collection(collection_name=self._collection)
            fields = description.get("fields") or []
            field_names = {
                str(field.get("name"))
                for field in fields
                if isinstance(field, dict) and field.get("name")
            }
            required_fields = {
                "memory_id",
                "scope",
                "user_id",
                "workspace_id",
                "kind",
                "memory_version",
                "vector",
            }
            missing_fields = required_fields - field_names
            if missing_fields:
                stats = self._client.get_collection_stats(collection_name=self._collection)
                row_count = int((stats or {}).get("row_count") or 0)
                if row_count:
                    raise ValueError(
                        f"Milvus collection {self._collection!r} misses fields "
                        f"{sorted(missing_fields)} and contains {row_count} rows"
                    )
                self._client.drop_collection(collection_name=self._collection)
                return self._ensure_collection_sync()
            vector_field = next(
                (
                    field
                    for field in fields
                    if isinstance(field, dict) and field.get("name") == "vector"
                ),
                None,
            )
            configured_dimension = field_dimension(vector_field)
            if configured_dimension is not None and configured_dimension != self._dimension:
                raise ValueError(
                    f"Milvus collection {self._collection!r} has dimension "
                    f"{configured_dimension}, expected {self._dimension}"
                )
            self._client.load_collection(collection_name=self._collection)
            return False

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description="Capsule structured Agent-memory text embeddings",
        )
        schema.add_field(
            field_name="memory_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(field_name="scope", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="workspace_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="memory_version", datatype=DataType.INT64)
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self._dimension,
        )
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_hnsw",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(collection_name=self._collection)
        return True

    def _delete_expression(self, expression: str) -> int:
        response = self._client.delete(
            collection_name=self._collection,
            filter=expression,
        )
        return delete_count(response)


def _memory_filter(
    *,
    scope: Literal["workspace", "global"],
    user_id: str,
    workspace_id: str,
) -> str:
    clauses = [
        f"scope == {json.dumps(scope)}",
        f"user_id == {json.dumps(user_id)}",
    ]
    if scope == "workspace":
        clauses.append(f"workspace_id == {json.dumps(workspace_id)}")
    else:
        clauses.append('workspace_id == ""')
    return " and ".join(clauses)


def _parse_hits(raw: object) -> list[AgentMemoryVectorHit]:
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        return []
    hits: list[AgentMemoryVectorHit] = []
    for item in raw[0]:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        memory_id = (
            entity.get("memory_id") or item.get("id")
            if isinstance(entity, dict)
            else item.get("id")
        )
        memory_version = entity.get("memory_version") if isinstance(entity, dict) else None
        distance = item.get("distance")
        if not isinstance(memory_id, str) or distance is None or memory_version is None:
            continue
        try:
            hits.append(
                AgentMemoryVectorHit(
                    memory_id=memory_id,
                    distance=float(distance),
                    memory_version=int(memory_version),
                )
            )
        except (TypeError, ValueError):
            continue
    return hits
