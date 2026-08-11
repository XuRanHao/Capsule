"""Build materialized search vectors without mutating raw embedding records.

The materialized records deliberately reuse the selected-dimension embedding
ID.  Callers must therefore provide a destination vector store backed by a
collection that is separate from the raw embedding collection.  Stable IDs
make retries idempotent in that destination collection.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from capsule.enums import EmbeddingType
from capsule.pipeline.vector_fusion import (
    DEFAULT_NATIVE_CONTENT_WEIGHT,
    fuse_native_dimension_vectors,
    validate_native_content_weight,
)
from capsule.vectorstore.milvus import VectorRecord


class SearchVectorStore(Protocol):
    """The write-only destination required by the materializer."""

    async def ensure_collection(self) -> bool: ...

    async def aupsert_fused(self, records: list[VectorRecord]) -> None: ...


@dataclass(slots=True, frozen=True)
class SearchVectorMaterializationFailure:
    asset_id: str
    embedding_id: str
    reason: str


@dataclass(slots=True, frozen=True)
class SearchVectorMaterializationResult:
    requested_count: int
    materialized_count: int
    embedding_ids: tuple[str, ...]
    failures: tuple[SearchVectorMaterializationFailure, ...] = ()

    @property
    def failed_count(self) -> int:
        return len(self.failures)


class SearchVectorIndexMaterializer:
    """Fuse current raw records and upsert them into an isolated search index."""

    def __init__(
        self,
        vector_store: SearchVectorStore,
        *,
        native_content_weight: float = DEFAULT_NATIVE_CONTENT_WEIGHT,
    ) -> None:
        self._vector_store = vector_store
        self._native_content_weight = validate_native_content_weight(
            native_content_weight
        )

    async def materialize(
        self,
        *,
        native_records: Sequence[VectorRecord],
        dimension_records: Sequence[VectorRecord],
    ) -> SearchVectorMaterializationResult:
        """Materialize every dimension record that has one matching native record.

        Inputs are expected to be the repository-selected current records.  A
        missing or incompatible pair is reported without preventing independent
        Assets from being indexed.  Destination write errors are allowed to
        propagate so the whole stable-ID batch can be retried.
        """

        native_by_asset = _index_native_records(native_records)
        _validate_unique_dimension_ids(dimension_records)
        derived: list[VectorRecord] = []
        failures: list[SearchVectorMaterializationFailure] = []
        for dimension_record in dimension_records:
            native_record = native_by_asset.get(dimension_record.asset_id)
            if native_record is None:
                failures.append(
                    SearchVectorMaterializationFailure(
                        asset_id=dimension_record.asset_id,
                        embedding_id=dimension_record.embedding_id,
                        reason="current native_multimodal embedding is missing",
                    )
                )
                continue
            try:
                derived.append(
                    build_fused_search_vector_record(
                        native_record=native_record,
                        dimension_record=dimension_record,
                        native_content_weight=self._native_content_weight,
                    )
                )
            except ValueError as exc:
                failures.append(
                    SearchVectorMaterializationFailure(
                        asset_id=dimension_record.asset_id,
                        embedding_id=dimension_record.embedding_id,
                        reason=str(exc),
                    )
                )

        if derived:
            await self._vector_store.ensure_collection()
            await self._vector_store.aupsert_fused(derived)
        return SearchVectorMaterializationResult(
            requested_count=len(dimension_records),
            materialized_count=len(derived),
            embedding_ids=tuple(record.embedding_id for record in derived),
            failures=tuple(failures),
        )


def build_fused_search_vector_record(
    *,
    native_record: VectorRecord,
    dimension_record: VectorRecord,
    native_content_weight: float = DEFAULT_NATIVE_CONTENT_WEIGHT,
) -> VectorRecord:
    """Return a fused record carrying the dimension record's stable identity."""

    _validate_pair(native_record=native_record, dimension_record=dimension_record)
    vector = fuse_native_dimension_vectors(
        native_vector=native_record.vector,
        dimension_vector=dimension_record.vector,
        native_content_weight=native_content_weight,
    )
    return replace(dimension_record, vector=vector)


def _index_native_records(
    records: Sequence[VectorRecord],
) -> dict[str, VectorRecord]:
    indexed: dict[str, VectorRecord] = {}
    for record in records:
        if record.embedding_type != EmbeddingType.NATIVE_MULTIMODAL.value:
            raise ValueError(
                "native_records must contain only native_multimodal embeddings"
            )
        if record.asset_id in indexed:
            raise ValueError(
                f"multiple current native embeddings provided for asset {record.asset_id}"
            )
        indexed[record.asset_id] = record
    return indexed


def _validate_unique_dimension_ids(records: Sequence[VectorRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.embedding_id in seen:
            raise ValueError(
                f"duplicate dimension embedding_id provided: {record.embedding_id}"
            )
        seen.add(record.embedding_id)


def _validate_pair(
    *,
    native_record: VectorRecord,
    dimension_record: VectorRecord,
) -> None:
    if native_record.embedding_type != EmbeddingType.NATIVE_MULTIMODAL.value:
        raise ValueError("native record must use native_multimodal embedding_type")
    if dimension_record.embedding_type == EmbeddingType.NATIVE_MULTIMODAL.value:
        raise ValueError("dimension record must use a non-native embedding_type")
    identity_fields = (
        "asset_id",
        "workspace_id",
        "project_id",
        "source_file_id",
        "asset_type",
        "file_type",
        "model_name",
        "embedding_revision",
    )
    mismatches = [
        field_name
        for field_name in identity_fields
        if getattr(native_record, field_name) != getattr(dimension_record, field_name)
    ]
    if mismatches:
        raise ValueError(
            "native and dimension metadata mismatch: " + ", ".join(mismatches)
        )
