import math
from dataclasses import dataclass

from pymilvus import MilvusClient

from capsule.config import Settings


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
    def __init__(self, settings: Settings) -> None:
        token = settings.milvus_token.get_secret_value() if settings.milvus_token else ""
        self._client = MilvusClient(uri=settings.milvus_uri, token=token)
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
