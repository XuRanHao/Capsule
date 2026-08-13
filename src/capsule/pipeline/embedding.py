"""Generate Asset Embeddings and durably index them in PostgreSQL and Milvus.

This module owns only one channel at a time.  It deliberately does not create
descriptions or features: when those fields are later populated, the same
service can index their distinct ``EmbeddingType`` values without mixing them
with ``native_multimodal`` vectors.
"""

import asyncio
import base64
import hashlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import EmbeddingAsset, EmbeddingRepository
from capsule.enums import AssetType, EmbeddingSourceMode, EmbeddingType
from capsule.features import (
    asset_usage_embedding_text,
    effective_feature_text,
    embedding_type_supports_asset_type,
)
from capsule.media.model_image import ModelImageCache
from capsule.media.video_frames import (
    FFmpegVideoFrameExtractor,
    VideoFrameExtractor,
    logical_video_frame_request,
)
from capsule.pipeline.search_vector_index import (
    SearchVectorIndexMaterializer,
    SearchVectorMaterializationFailure,
    SearchVectorMaterializationResult,
)
from capsule.schemas import EmbeddingResult
from capsule.vectorstore.milvus import VectorRecord
from capsule.video_output import is_logical_video_asset

logger = logging.getLogger(__name__)


class AssetEmbeddingClient(Protocol):
    async def embed_multimodal(
        self,
        input_items: Sequence[Mapping[str, Any]],
    ) -> EmbeddingResult: ...


class EmbeddingVectorStore(Protocol):
    async def ensure_collection(self) -> bool: ...

    async def aupsert(self, records: list[VectorRecord]) -> None: ...

    async def aupsert_fused(self, records: list[VectorRecord]) -> None: ...

    async def fetch_vectors(self, embedding_ids: Sequence[str]) -> dict[str, list[float]]: ...

    async def fetch_fused_vector_ids(self, embedding_ids: Sequence[str]) -> set[str]: ...


class ArtifactReader(Protocol):
    async def download_uri(self, uri: str) -> bytes: ...


class EmbeddingRunResult(BaseModel):
    workspace_id: str
    embedding_type: str
    requested_asset_count: int
    indexed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    embedding_duration_ms: float = 0
    indexing_duration_ms: float = 0
    embedding_ids: list[str] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _EmbeddingInput:
    input_items: list[dict[str, Any]]
    source_content_hash: str
    source_mode: EmbeddingSourceMode


@dataclass(slots=True, frozen=True)
class _EmbeddingOutcome:
    kind: str
    asset_id: str
    embedding_id: str | None = None
    error: str | None = None
    model_duration_ms: float = 0
    indexing_duration_ms: float = 0


class EmbeddingInputUnavailable(ValueError):
    """An Asset has no material for the requested Embedding channel yet."""


# ===========================================
#      Asset → EmbeddingRecord → Milvus
# ===========================================


class AssetEmbeddingService:
    """Index one Embedding channel with recoverable per-Asset failures.

    A PostgreSQL record moves to ``processing`` before the model call.  Milvus
    uses that record's stable ID as its primary key, so a retry safely upserts
    instead of creating a second vector.  If the final PostgreSQL update fails,
    the next run reuses the same primary key and repairs the state.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        repository: EmbeddingRepository,
        model_client: AssetEmbeddingClient,
        vector_store: EmbeddingVectorStore,
        artifact_reader: ArtifactReader | None = None,
        image_cache: ModelImageCache | None = None,
        video_frame_extractor: VideoFrameExtractor | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._model_client = model_client
        self._vector_store = vector_store
        self._search_vector_materializer = SearchVectorIndexMaterializer(vector_store)
        self._artifact_reader = artifact_reader
        self._image_cache = image_cache or ModelImageCache(
            target_bytes=settings.model_image_target_bytes,
            max_edge=settings.model_image_max_edge,
            max_entries=settings.model_image_cache_entries,
        )
        self._video_frame_extractor = video_frame_extractor or FFmpegVideoFrameExtractor(
            concurrency=settings.ffmpeg_concurrency
        )
        self._native_semaphore = asyncio.Semaphore(settings.native_embedding_concurrency)
        self._video_native_semaphore = asyncio.Semaphore(
            settings.video_native_embedding_concurrency
        )
        self._text_semaphore = asyncio.Semaphore(settings.embedding_concurrency)
        self._collection_ready = False
        self._collection_lock = asyncio.Lock()
        self._search_vector_locks: dict[tuple[str, EmbeddingType], asyncio.Lock] = {}
        self._search_vector_source_fingerprints: dict[
            tuple[str, EmbeddingType], str
        ] = {}

    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL,
        asset_ids: Sequence[str] | None = None,
        force: bool = False,
    ) -> EmbeddingRunResult:
        """Generate one independent vector per eligible Asset in a workspace."""
        assets = await self._repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        if assets:
            await self._ensure_collection()
        return await self._run_assets(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            assets=assets,
            force=force,
        )

    async def run_many(
        self,
        *,
        workspace_id: str,
        embedding_types: Sequence[EmbeddingType],
        asset_ids: Sequence[str] | None = None,
        force: bool = False,
    ) -> list[EmbeddingRunResult]:
        """Run independent channels concurrently behind shared workload pools."""
        selected_types = list(dict.fromkeys(embedding_types))
        if not selected_types:
            return []
        assets = await self._repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        if assets:
            await self._ensure_collection()
        return list(
            await asyncio.gather(
                *(
                    self._run_assets(
                        workspace_id=workspace_id,
                        embedding_type=embedding_type,
                        assets=assets,
                        force=force,
                    )
                    for embedding_type in selected_types
                )
            )
        )

    async def materialize_search_vectors(
        self,
        *,
        workspace_id: str,
        embedding_types: Sequence[EmbeddingType],
        asset_ids: Sequence[str] | None = None,
    ) -> SearchVectorMaterializationResult:
        """Build current non-native search vectors from already indexed raw pairs."""
        selected_types = tuple(
            embedding_type
            for embedding_type in dict.fromkeys(embedding_types)
            if embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
        )
        if not selected_types or (asset_ids is not None and not asset_ids):
            return SearchVectorMaterializationResult(
                requested_count=0,
                materialized_count=0,
                embedding_ids=(),
            )

        assets = await self._repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        assets_by_id = {asset.asset_id: asset for asset in assets}
        if not assets_by_id:
            return SearchVectorMaterializationResult(
                requested_count=0,
                materialized_count=0,
                embedding_ids=(),
            )

        native_embeddings, *dimension_embedding_groups = await asyncio.gather(
            self._repository.list_indexed_cluster_embeddings(
                workspace_id=workspace_id,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
                model_name=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
                milvus_collection=self._settings.milvus_collection,
                asset_ids=tuple(assets_by_id),
            ),
            *(
                self._repository.list_indexed_cluster_embeddings(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type.value,
                    model_name=self._settings.embedding_model,
                    dimension=self._settings.embedding_dimension,
                    milvus_collection=self._settings.milvus_collection,
                    asset_ids=tuple(assets_by_id),
                )
                for embedding_type in selected_types
            ),
        )
        native_embeddings = [
            item for item in native_embeddings if item.asset_id in assets_by_id
        ]
        dimension_embeddings = [
            (embedding_type, item)
            for embedding_type, group in zip(
                selected_types,
                dimension_embedding_groups,
                strict=True,
            )
            for item in group
            if item.asset_id in assets_by_id
        ]
        dimension_asset_ids = {item.asset_id for _, item in dimension_embeddings}
        native_embeddings = [
            item for item in native_embeddings if item.asset_id in dimension_asset_ids
        ]
        if not dimension_embeddings:
            return SearchVectorMaterializationResult(
                requested_count=0,
                materialized_count=0,
                embedding_ids=(),
            )

        await self._ensure_collection()
        raw_embedding_ids = {
            item.embedding_id for item in native_embeddings
        } | {
            item.embedding_id for _, item in dimension_embeddings
        }
        raw_vectors = await self._fetch_visible_vectors(raw_embedding_ids)

        native_records = [
            _current_vector_record(
                asset=assets_by_id[item.asset_id],
                embedding_id=item.embedding_id,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                model_name=self._settings.embedding_model,
                vector=raw_vectors[item.embedding_id],
            )
            for item in native_embeddings
            if item.embedding_id in raw_vectors
        ]
        dimension_records: list[VectorRecord] = []
        preparation_failures: list[SearchVectorMaterializationFailure] = []
        for embedding_type, item in dimension_embeddings:
            vector = raw_vectors.get(item.embedding_id)
            if vector is None:
                preparation_failures.append(
                    SearchVectorMaterializationFailure(
                        asset_id=item.asset_id,
                        embedding_id=item.embedding_id,
                        reason="current dimension raw vector is missing",
                    )
                )
                continue
            dimension_records.append(
                _current_vector_record(
                    asset=assets_by_id[item.asset_id],
                    embedding_id=item.embedding_id,
                    embedding_type=embedding_type,
                    model_name=self._settings.embedding_model,
                    vector=vector,
                )
            )

        materialized = await self._search_vector_materializer.materialize(
            native_records=native_records,
            dimension_records=dimension_records,
        )
        return SearchVectorMaterializationResult(
            requested_count=len(dimension_embeddings),
            materialized_count=materialized.materialized_count,
            embedding_ids=materialized.embedding_ids,
            failures=(*preparation_failures, *materialized.failures),
        )

    async def ensure_search_vectors(
        self,
        *,
        workspace_id: str,
        embedding_types: Sequence[EmbeddingType],
    ) -> dict[EmbeddingType, str]:
        """Lazily materialize changed non-native search channels once per service."""
        selected_types = tuple(
            embedding_type
            for embedding_type in dict.fromkeys(embedding_types)
            if embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
        )
        outcomes = await asyncio.gather(
            *(
                self._ensure_search_vector_dimension(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type,
                )
                for embedding_type in selected_types
            )
        )
        return {
            embedding_type: error
            for embedding_type, error in zip(selected_types, outcomes, strict=True)
            if error is not None
        }

    async def _ensure_search_vector_dimension(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
    ) -> str | None:
        cache_key = (workspace_id, embedding_type)
        lock = self._search_vector_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            try:
                fingerprint, expected_ids = await self._search_vector_source_state(
                    workspace_id=workspace_id,
                    embedding_type=embedding_type,
                )
                if self._search_vector_source_fingerprints.get(cache_key) == fingerprint:
                    return None
                visible_ids = await self._vector_store.fetch_fused_vector_ids(expected_ids)
                if visible_ids == set(expected_ids):
                    self._search_vector_source_fingerprints[cache_key] = fingerprint
                    return None
                result = await self.materialize_search_vectors(
                    workspace_id=workspace_id,
                    embedding_types=[embedding_type],
                )
            except Exception as exc:
                logger.exception(
                    "search-vector materialization failed workspace=%s dimension=%s",
                    workspace_id,
                    embedding_type.value,
                )
                return str(exc) or type(exc).__name__
            if result.failures:
                return _materialization_failure_message(result)
            self._search_vector_source_fingerprints[cache_key] = fingerprint
            return None

    async def _search_vector_source_state(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
    ) -> tuple[str, tuple[str, ...]]:
        native_embeddings, dimension_embeddings = await asyncio.gather(
            self._repository.list_indexed_cluster_embeddings(
                workspace_id=workspace_id,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
                model_name=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
                milvus_collection=self._settings.milvus_collection,
            ),
            self._repository.list_indexed_cluster_embeddings(
                workspace_id=workspace_id,
                embedding_type=embedding_type.value,
                model_name=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
                milvus_collection=self._settings.milvus_collection,
            ),
        )
        dimension_asset_ids = {item.asset_id for item in dimension_embeddings}
        source_ids = [
            *(
                f"native:{item.asset_id}:{item.embedding_id}"
                for item in native_embeddings
                if item.asset_id in dimension_asset_ids
            ),
            *(
                f"dimension:{item.asset_id}:{item.embedding_id}"
                for item in dimension_embeddings
            ),
        ]
        encoded = "\n".join(sorted(source_ids)).encode("utf-8")
        expected_ids = tuple(sorted(item.embedding_id for item in dimension_embeddings))
        return hashlib.sha256(encoded).hexdigest(), expected_ids

    async def _fetch_visible_vectors(
        self,
        embedding_ids: set[str],
    ) -> dict[str, list[float]]:
        remaining = set(embedding_ids)
        vectors: dict[str, list[float]] = {}
        deadline = (
            asyncio.get_running_loop().time()
            + self._settings.search_vector_visibility_timeout_seconds
        )
        delay = self._settings.search_vector_visibility_poll_initial_seconds
        while remaining:
            fetched = await self._vector_store.fetch_vectors(sorted(remaining))
            for embedding_id in remaining.intersection(fetched):
                vectors[embedding_id] = fetched[embedding_id]
            remaining.difference_update(fetched)
            if not remaining:
                break
            now = asyncio.get_running_loop().time()
            if now >= deadline:
                break
            await asyncio.sleep(min(delay, deadline - now))
            delay = min(
                delay * 2,
                self._settings.search_vector_visibility_poll_max_seconds,
            )
        return vectors

    async def _ensure_collection(self) -> None:
        """Initialize Milvus once even when streaming Assets arrive concurrently."""
        if self._collection_ready:
            return
        async with self._collection_lock:
            if self._collection_ready:
                return
            await self._vector_store.ensure_collection()
            self._collection_ready = True

    async def _run_assets(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        assets: Sequence[EmbeddingAsset],
        force: bool,
    ) -> EmbeddingRunResult:
        run_started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(
                self._embed_one(
                    asset=asset,
                    embedding_type=embedding_type,
                    force=force,
                )
                for asset in assets
            )
        )
        elapsed_ms = (time.perf_counter() - run_started) * 1000 if assets else 0.0
        model_ms = sum(outcome.model_duration_ms for outcome in outcomes)
        indexing_ms = sum(outcome.indexing_duration_ms for outcome in outcomes)
        measured_ms = model_ms + indexing_ms
        embedding_ms = elapsed_ms * model_ms / measured_ms if measured_ms else elapsed_ms
        return EmbeddingRunResult(
            workspace_id=workspace_id,
            embedding_type=embedding_type.value,
            requested_asset_count=len(assets),
            indexed_count=sum(outcome.kind == "indexed" for outcome in outcomes),
            skipped_count=sum(outcome.kind == "skipped" for outcome in outcomes),
            failed_count=sum(outcome.kind == "failed" for outcome in outcomes),
            embedding_duration_ms=embedding_ms,
            indexing_duration_ms=max(0.0, elapsed_ms - embedding_ms),
            embedding_ids=[
                outcome.embedding_id
                for outcome in outcomes
                if outcome.kind == "indexed" and outcome.embedding_id is not None
            ],
            errors=[
                {"asset_id": outcome.asset_id, "error": outcome.error or "embedding failed"}
                for outcome in outcomes
                if outcome.kind == "failed"
            ],
        )

    async def _embed_one(
        self,
        *,
        asset: EmbeddingAsset,
        embedding_type: EmbeddingType,
        force: bool,
    ) -> _EmbeddingOutcome:
        if (
            embedding_type is EmbeddingType.NATIVE_MULTIMODAL
            and asset.asset_type == AssetType.VIDEO_SEGMENT.value
        ):
            async with self._video_native_semaphore:
                return await self._embed_one_with_shared_pool(
                    asset=asset,
                    embedding_type=embedding_type,
                    force=force,
                )
        return await self._embed_one_with_shared_pool(
            asset=asset,
            embedding_type=embedding_type,
            force=force,
        )

    async def _embed_one_with_shared_pool(
        self,
        *,
        asset: EmbeddingAsset,
        embedding_type: EmbeddingType,
        force: bool,
    ) -> _EmbeddingOutcome:
        semaphore = (
            self._native_semaphore
            if embedding_type is EmbeddingType.NATIVE_MULTIMODAL
            else self._text_semaphore
        )
        async with semaphore:
            prepared_id: str | None = None
            model_duration_ms = 0.0
            indexing_duration_ms = 0.0
            try:
                embedding_input = await self._build_input(asset, embedding_type)
                prepared = await self._repository.prepare(
                    asset=asset,
                    embedding_type=embedding_type.value,
                    model_name=self._settings.embedding_model,
                    dimension=self._settings.embedding_dimension,
                    source_content_hash=embedding_input.source_content_hash,
                    source_mode=embedding_input.source_mode.value,
                    milvus_collection=self._settings.milvus_collection,
                    force=force,
                )
                prepared_id = prepared.embedding_id
                if prepared.already_indexed:
                    return _EmbeddingOutcome(
                        kind="skipped",
                        asset_id=asset.asset_id,
                        embedding_id=prepared.embedding_id,
                    )

                phase_started = time.perf_counter()
                try:
                    response = await self._model_client.embed_multimodal(
                        embedding_input.input_items
                    )
                    # Release transient image/video payloads before vector persistence.
                    del embedding_input
                finally:
                    model_duration_ms = (time.perf_counter() - phase_started) * 1000
                latency_ms = round(model_duration_ms)
                phase_started = time.perf_counter()
                try:
                    await self._vector_store.aupsert(
                        [
                            VectorRecord(
                                embedding_id=prepared.milvus_primary_key,
                                workspace_id=asset.workspace_id,
                                project_id=asset.project_id,
                                asset_id=asset.asset_id,
                                source_file_id=asset.source_file_id,
                                asset_type=asset.asset_type,
                                file_type=asset.file_type,
                                embedding_type=embedding_type.value,
                                model_name=self._settings.embedding_model,
                                embedding_revision=asset.embedding_revision,
                                created_at_ts=int(asset.created_at.timestamp()),
                                vector=response.vector,
                            )
                        ]
                    )
                    await self._repository.mark_indexed(
                        embedding_id=prepared.embedding_id,
                        latency_ms=latency_ms,
                        usage=response.usage,
                    )
                finally:
                    indexing_duration_ms = (time.perf_counter() - phase_started) * 1000
                return _EmbeddingOutcome(
                    kind="indexed",
                    asset_id=asset.asset_id,
                    embedding_id=prepared.embedding_id,
                    model_duration_ms=model_duration_ms,
                    indexing_duration_ms=indexing_duration_ms,
                )
            except EmbeddingInputUnavailable:
                return _EmbeddingOutcome(kind="skipped", asset_id=asset.asset_id)
            except Exception as exc:
                if prepared_id is not None:
                    try:
                        await self._repository.mark_failed(embedding_id=prepared_id)
                    except Exception:
                        logger.exception(
                            "could not record embedding failure for asset %s",
                            asset.asset_id,
                        )
                error = str(exc) or type(exc).__name__
                logger.exception("embedding failed for asset %s", asset.asset_id)
                return _EmbeddingOutcome(
                    kind="failed",
                    asset_id=asset.asset_id,
                    embedding_id=prepared_id,
                    error=error[:2000],
                    model_duration_ms=model_duration_ms,
                    indexing_duration_ms=indexing_duration_ms,
                )

    async def _build_input(
        self,
        asset: EmbeddingAsset,
        embedding_type: EmbeddingType,
    ) -> _EmbeddingInput:
        if not embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=asset.asset_type,
        ):
            raise EmbeddingInputUnavailable(
                f"{embedding_type.value} is not supported for {asset.asset_type}"
            )
        if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
            return await self._native_input(asset)
        if embedding_type is EmbeddingType.ASSET_DESCRIPTION:
            return _text_input(
                asset.asset_description,
                embedding_type=embedding_type,
                source_mode=EmbeddingSourceMode.DESCRIPTION_TEXT,
            )
        if embedding_type is EmbeddingType.ASSET_USAGE:
            return _text_input(
                asset_usage_embedding_text(
                    asset.asset_features,
                    asset.source_relative_path,
                ),
                embedding_type=embedding_type,
                source_mode=EmbeddingSourceMode.FEATURE_TEXT,
            )
        return _text_input(
            effective_feature_text(asset.asset_features, embedding_type),
            embedding_type=embedding_type,
            source_mode=EmbeddingSourceMode.FEATURE_TEXT,
        )

    async def _native_input(self, asset: EmbeddingAsset) -> _EmbeddingInput:
        if asset.asset_type in {
            AssetType.MARKDOWN_BLOCK.value,
            AssetType.TEXT_BLOCK.value,
            AssetType.AUDIO_SEGMENT.value,
        }:
            return _text_input(
                asset.raw_content,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                source_mode=EmbeddingSourceMode.ORIGINAL_TEXT,
            )
        if asset.asset_type == AssetType.IMAGE.value:
            image_uri = _image_source_uri(asset)
            prepared_image = await self._image_cache.prepare(
                cache_key=asset.content_hash,
                mime_type=_image_mime_type(asset),
                loader=lambda: _read_local_source(image_uri),
            )
            image_url = _data_uri(
                mime_type=prepared_image.mime_type,
                content=prepared_image.content,
            )
            input_items: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
            if asset.raw_content and asset.raw_content.strip():
                input_items.append(
                    {"type": "text", "text": f"图中文字：\n{asset.raw_content[:8_000]}"}
                )
            return _EmbeddingInput(
                input_items=input_items,
                source_content_hash=_hash_bytes(
                    EmbeddingType.NATIVE_MULTIMODAL.value.encode("utf-8"),
                    EmbeddingSourceMode.ORIGINAL_IMAGE.value.encode("utf-8"),
                    prepared_image.content,
                    (asset.raw_content or "").encode("utf-8"),
                ),
                source_mode=EmbeddingSourceMode.ORIGINAL_IMAGE,
            )
        if asset.asset_type == AssetType.VIDEO_SEGMENT.value:
            if is_logical_video_asset(
                asset_type=asset.asset_type,
                file_info=asset.file_info,
            ):
                frames = await self._video_frame_extractor.extract(
                    logical_video_frame_request(
                        source_uri=asset.source_storage_uri,
                        file_info=asset.file_info,
                        source_locator=asset.source_locator,
                    )
                )
                if not frames:
                    raise EmbeddingInputUnavailable(
                        "logical video Asset has no readable representative frame"
                    )
                return _EmbeddingInput(
                    input_items=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _data_uri(mime_type="image/jpeg", content=frame)
                            },
                        }
                        for frame in frames
                    ],
                    source_content_hash=_hash_bytes(
                        EmbeddingType.NATIVE_MULTIMODAL.value.encode("utf-8"),
                        EmbeddingSourceMode.ORIGINAL_VIDEO.value.encode("utf-8"),
                        *frames,
                    ),
                    source_mode=EmbeddingSourceMode.ORIGINAL_VIDEO,
                )
            video_data_uri = await self._video_data_uri(asset)
            return _EmbeddingInput(
                input_items=[{"type": "video_url", "video_url": {"url": video_data_uri}}],
                source_content_hash=_hash_text(
                    EmbeddingType.NATIVE_MULTIMODAL.value,
                    EmbeddingSourceMode.ORIGINAL_VIDEO.value,
                    asset.content_hash,
                ),
                source_mode=EmbeddingSourceMode.ORIGINAL_VIDEO,
            )
        raise EmbeddingInputUnavailable(f"unsupported native Asset type: {asset.asset_type}")

    async def _video_data_uri(self, asset: EmbeddingAsset) -> str:
        if not asset.derived_file_uri:
            raise EmbeddingInputUnavailable("video Asset has no derived MP4")
        parsed = urlparse(asset.derived_file_uri)
        if parsed.scheme == "data":
            return asset.derived_file_uri
        if parsed.scheme != "s3" or self._artifact_reader is None:
            raise EmbeddingInputUnavailable(
                "video Asset needs a data URI or configured object storage reader"
            )
        content = await self._artifact_reader.download_uri(asset.derived_file_uri)
        return _data_uri(mime_type="video/mp4", content=content)


def _current_vector_record(
    *,
    asset: EmbeddingAsset,
    embedding_id: str,
    embedding_type: EmbeddingType,
    model_name: str,
    vector: list[float],
) -> VectorRecord:
    return VectorRecord(
        embedding_id=embedding_id,
        workspace_id=asset.workspace_id,
        project_id=asset.project_id,
        asset_id=asset.asset_id,
        source_file_id=asset.source_file_id,
        asset_type=asset.asset_type,
        file_type=asset.file_type,
        embedding_type=embedding_type.value,
        model_name=model_name,
        embedding_revision=asset.embedding_revision,
        created_at_ts=int(asset.created_at.timestamp()),
        vector=vector,
    )


def _materialization_failure_message(
    result: SearchVectorMaterializationResult,
) -> str:
    return "; ".join(
        f"{failure.asset_id}/{failure.embedding_id}: {failure.reason}"
        for failure in result.failures
    )


def _text_input(
    text: str | None,
    *,
    embedding_type: EmbeddingType,
    source_mode: EmbeddingSourceMode,
) -> _EmbeddingInput:
    if text is None or not text.strip():
        raise EmbeddingInputUnavailable(f"Asset has no content for {embedding_type.value}")
    return _EmbeddingInput(
        input_items=[{"type": "text", "text": text}],
        source_content_hash=_hash_text(embedding_type.value, source_mode.value, text),
        source_mode=source_mode,
    )


def _read_local_source(storage_uri: str) -> bytes:
    parsed = urlparse(storage_uri)
    if parsed.scheme != "file":
        raise EmbeddingInputUnavailable(
            f"image source must be a local file URI, got {parsed.scheme or 'no scheme'}"
        )
    path = Path(unquote(parsed.path))
    if not path.is_file():
        raise EmbeddingInputUnavailable(f"image source file no longer exists: {path}")
    return path.read_bytes()


def _image_source_uri(asset: EmbeddingAsset) -> str:
    """Use a materialised document image instead of its container source file."""

    return asset.derived_file_uri or asset.source_storage_uri


def _image_mime_type(asset: EmbeddingAsset) -> str:
    value = asset.file_info.get("mime_type")
    return value if isinstance(value, str) and value else asset.source_mime_type


def _data_uri(*, mime_type: str, content: bytes) -> str:
    if not content:
        raise EmbeddingInputUnavailable("model media is empty")
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _hash_text(*parts: str) -> str:
    return _hash_bytes(*(part.encode("utf-8") for part in parts))


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, byteorder="big"))
        digest.update(part)
    return digest.hexdigest()
