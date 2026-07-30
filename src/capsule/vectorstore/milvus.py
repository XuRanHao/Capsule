import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient

from capsule.config import Settings
from capsule.search.models import SearchFilters, VectorSearchHit


@dataclass(slots=True, frozen=True)
class VectorRecord:
    embedding_id: str
    workspace_id: str
    project_id: str
    asset_id: str
    source_file_id: str
    asset_type: str
    file_type: str
    embedding_type: str
    model_name: str
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
        self._search_ef = settings.search_hnsw_ef

    async def ensure_collection(self) -> bool:
        """Create and load the frozen Seed-1.6 collection when it is missing.

        Returns ``True`` when a collection was created and ``False`` when the
        existing collection already matched the configured vector dimension.
        """

        return await asyncio.to_thread(self._ensure_collection_sync)

    def _ensure_collection_sync(self) -> bool:
        if self._client.has_collection(collection_name=self._collection):
            description = self._client.describe_collection(
                collection_name=self._collection,
            )
            fields = description.get("fields") or []
            field_names = {
                str(field.get("name"))
                for field in fields
                if isinstance(field, dict) and field.get("name")
            }
            required_fields = {
                "embedding_id",
                "workspace_id",
                "project_id",
                "asset_id",
                "source_file_id",
                "asset_type",
                "file_type",
                "embedding_type",
                "model_name",
                "embedding_revision",
                "created_at_ts",
                "vector",
            }
            missing_fields = required_fields - field_names
            if missing_fields:
                stats = self._client.get_collection_stats(
                    collection_name=self._collection,
                )
                row_count = int((stats or {}).get("row_count") or 0)
                if row_count:
                    raise ValueError(
                        f"Milvus collection {self._collection!r} misses fields "
                        f"{sorted(missing_fields)} and contains {row_count} rows; "
                        "migrate it before starting search"
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
            configured_dimension = _field_dimension(vector_field)
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
            description="Capsule multimodal asset embeddings",
        )
        schema.add_field(
            field_name="embedding_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=64,
        )
        schema.add_field(
            field_name="workspace_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="project_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="asset_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="source_file_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="asset_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(
            field_name="file_type",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(
            field_name="embedding_type",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="model_name",
            datatype=DataType.VARCHAR,
            max_length=255,
        )
        schema.add_field(
            field_name="embedding_revision",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="created_at_ts",
            datatype=DataType.INT64,
        )
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
                        "project_id": record.project_id,
                        "asset_id": record.asset_id,
                        "source_file_id": record.source_file_id,
                        "asset_type": record.asset_type,
                        "file_type": record.file_type,
                        "embedding_type": record.embedding_type,
                        "model_name": record.model_name,
                        "embedding_revision": record.embedding_revision,
                        "created_at_ts": record.created_at_ts,
                        "vector": record.vector,
                    }
                    for record in batch
                ],
            )

    async def aupsert(self, records: list[VectorRecord]) -> None:
        """Write vectors without blocking callers that process many Assets."""
        await asyncio.to_thread(self.upsert, records)

    async def search(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[VectorSearchHit]:
        """Search one embedding channel without blocking the event loop."""
        self.validate_vector(vector)
        expression = self.build_filter_expression(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            filters=filters,
        )
        raw = await asyncio.to_thread(
            self._client.search,
            collection_name=self._collection,
            data=[vector],
            anns_field="vector",
            filter=expression,
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {"ef": self._search_ef}},
            output_fields=[
                "embedding_id",
                "asset_id",
                "source_file_id",
                "asset_type",
                "embedding_revision",
            ],
        )
        return _parse_search_hits(raw)

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]:
        """Fetch exact vectors by embedding primary key for offline clustering."""
        if not embedding_ids:
            return {}
        return await asyncio.to_thread(self._fetch_vectors_sync, list(embedding_ids))

    async def delete_workspace(self, workspace_id: str) -> int:
        """Remove every vector belonging to one cleared workspace."""
        return await asyncio.to_thread(self._delete_workspace_sync, workspace_id)

    def _delete_workspace_sync(self, workspace_id: str) -> int:
        response = self._client.delete(
            collection_name=self._collection,
            filter=f"workspace_id == {json.dumps(workspace_id)}",
        )
        if not isinstance(response, dict):
            return 0
        value = response.get("delete_count", response.get("delete_cnt", 0))
        if not isinstance(value, (int, float, str)):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def delete_all(self) -> int:
        """Remove all Capsule vectors from the configured collection."""
        return await asyncio.to_thread(self._delete_all_sync)

    def _delete_all_sync(self) -> int:
        response = self._client.delete(
            collection_name=self._collection,
            filter='embedding_id != ""',
        )
        if not isinstance(response, dict):
            return 0
        value = response.get("delete_count", response.get("delete_cnt", 0))
        if not isinstance(value, (int, float, str)):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _fetch_vectors_sync(self, embedding_ids: list[str]) -> dict[str, list[float]]:
        vectors: dict[str, list[float]] = {}
        for start in range(0, len(embedding_ids), self._batch_size):
            batch = embedding_ids[start : start + self._batch_size]
            encoded_ids = ", ".join(json.dumps(item) for item in batch)
            rows = self._client.query(
                collection_name=self._collection,
                filter=f"embedding_id in [{encoded_ids}]",
                output_fields=["embedding_id", "vector"],
                limit=len(batch),
            )
            for row in rows:
                if not isinstance(row, dict):
                    continue
                embedding_id = row.get("embedding_id")
                vector = row.get("vector")
                if not isinstance(embedding_id, str) or not isinstance(vector, list):
                    continue
                try:
                    parsed = [float(value) for value in vector]
                    self.validate_vector(parsed)
                except (TypeError, ValueError):
                    continue
                vectors[embedding_id] = parsed
        return vectors

    @staticmethod
    def build_filter_expression(
        *,
        workspace_id: str,
        embedding_type: str,
        filters: SearchFilters | None = None,
    ) -> str:
        clauses = [
            f"workspace_id == {json.dumps(workspace_id)}",
            f"embedding_type == {json.dumps(embedding_type)}",
        ]
        filters = filters or SearchFilters()
        if filters.project_id:
            clauses.append(f"project_id == {json.dumps(filters.project_id)}")
        if filters.asset_type:
            encoded_types = ", ".join(json.dumps(item.value) for item in filters.asset_type)
            clauses.append(f"asset_type in [{encoded_types}]")
        if filters.file_type:
            encoded_file_types = ", ".join(json.dumps(item) for item in filters.file_type)
            clauses.append(f"file_type in [{encoded_file_types}]")
        if filters.source_file_id:
            encoded_sources = ", ".join(json.dumps(item) for item in filters.source_file_id)
            clauses.append(f"source_file_id in [{encoded_sources}]")
        if filters.model_name:
            encoded_model_names = ", ".join(json.dumps(item) for item in filters.model_name)
            clauses.append(f"model_name in [{encoded_model_names}]")
        if filters.created_at_from:
            clauses.append(f"created_at_ts >= {int(filters.created_at_from.timestamp())}")
        if filters.created_at_to:
            clauses.append(f"created_at_ts <= {int(filters.created_at_to.timestamp())}")
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


def _field_dimension(field: Any) -> int | None:
    if not isinstance(field, dict):
        return None
    params = field.get("params")
    if isinstance(params, dict) and params.get("dim") is not None:
        return int(params["dim"])
    if field.get("dim") is not None:
        return int(field["dim"])
    return None
