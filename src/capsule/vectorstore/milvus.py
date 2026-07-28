import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any

from pymilvus import MilvusClient

from capsule.config import Settings
from capsule.search.models import VectorSearchHit


@dataclass(slots=True, frozen=True)
class VectorRecord:
    embedding_id: str
    workspace_id: str
    asset_id: str
    source_file_id: str
    asset_type: str
    embedding_type: str
    model_version: str
    embedding_revision: int
    created_at_ts: int
    vector: list[float]


class MilvusVectorStore:
    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        token = settings.milvus_token.get_secret_value() if settings.milvus_token else ""
        self._client = client or MilvusClient(uri=settings.milvus_uri, token=token)
        self._collection = settings.milvus_collection
        self._dimension = settings.embedding_dimension
        self._batch_size = settings.milvus_batch_size

    def validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(f"expected {self._dimension} dimensions, got {len(vector)}")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("vector contains NaN or infinity")
        if not any(value != 0.0 for value in vector):
            raise ValueError("vector must not be all zeros")

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self.validate_vector(record.vector)
        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            self._client.upsert(
                collection_name=self._collection,
                data=[
                    {
                        "embedding_id": record.embedding_id,
                        "workspace_id": record.workspace_id,
                        "asset_id": record.asset_id,
                        "source_file_id": record.source_file_id,
                        "asset_type": record.asset_type,
                        "embedding_type": record.embedding_type,
                        "model_version": record.model_version,
                        "embedding_revision": record.embedding_revision,
                        "created_at_ts": record.created_at_ts,
                        "vector": record.vector,
                    }
                    for record in batch
                ],
            )

    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        asset_types: tuple[str, ...],
        limit: int,
    ) -> list[VectorSearchHit]:
        """Search one embedding channel without blocking the event loop."""
        self.validate_vector(vector)
        expression = self.build_filter_expression(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            asset_types=asset_types,
        )
        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            filter=expression,
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=[
                "embedding_id",
                "asset_id",
                "source_file_id",
                "asset_type",
                "embedding_revision",
            ],
        )
        return _parse_search_hits(raw)

    @staticmethod
    def build_filter_expression(
        *,
        workspace_id: str,
        embedding_type: str,
        asset_types: tuple[str, ...] = (),
    ) -> str:
        clauses = [
            f"workspace_id == {json.dumps(workspace_id)}",
            f"embedding_type == {json.dumps(embedding_type)}",
        ]
        if asset_types:
            encoded_types = ", ".join(json.dumps(item) for item in asset_types)
            clauses.append(f"asset_type in [{encoded_types}]")
        return " and ".join(clauses)


def _parse_search_hits(raw: Any) -> list[VectorSearchHit]:
    if not raw:
        return []
    hits = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    parsed: list[VectorSearchHit] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        entity = hit.get("entity") or {}
        if not isinstance(entity, dict):
            entity = {}
        embedding_id = str(entity.get("embedding_id") or hit.get("id") or "")
        asset_id = str(entity.get("asset_id") or "")
        if not embedding_id or not asset_id:
            continue
        parsed.append(
            VectorSearchHit(
                embedding_id=embedding_id,
                asset_id=asset_id,
                source_file_id=str(entity.get("source_file_id") or ""),
                asset_type=str(entity.get("asset_type") or ""),
                embedding_revision=int(entity.get("embedding_revision") or 1),
                similarity=float(hit.get("distance") or hit.get("score") or 0.0),
            )
        )
    return parsed
