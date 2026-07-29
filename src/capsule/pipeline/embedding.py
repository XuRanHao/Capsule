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
from capsule.schemas import EmbeddingResult
from capsule.vectorstore.milvus import VectorRecord

logger = logging.getLogger(__name__)


class AssetEmbeddingClient(Protocol):
    async def embed_multimodal(
        self,
        input_items: Sequence[Mapping[str, Any]],
    ) -> EmbeddingResult: ...


class EmbeddingVectorStore(Protocol):
    async def ensure_collection(self) -> bool: ...

    async def aupsert(self, records: list[VectorRecord]) -> None: ...


class VideoUrlSigner(Protocol):
    async def presigned_get_uri(self, uri: str, *, expires_seconds: int = 3600) -> str: ...


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


#===========================================
#      Asset → EmbeddingRecord → Milvus
#===========================================


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
        video_url_signer: VideoUrlSigner | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._model_client = model_client
        self._vector_store = vector_store
        self._video_url_signer = video_url_signer

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
        run_started = time.perf_counter()
        if assets:
            await self._vector_store.ensure_collection()

        semaphore = asyncio.Semaphore(self._settings.embedding_concurrency)
        outcomes = await asyncio.gather(
            *(
                self._embed_one(
                    asset=asset,
                    embedding_type=embedding_type,
                    force=force,
                    semaphore=semaphore,
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
        semaphore: asyncio.Semaphore,
    ) -> _EmbeddingOutcome:
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
                finally:
                    model_duration_ms = (
                        time.perf_counter() - phase_started
                    ) * 1000
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
                    indexing_duration_ms = (
                        time.perf_counter() - phase_started
                    ) * 1000
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
        if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
            return await self._native_input(asset)
        if embedding_type is EmbeddingType.ASSET_DESCRIPTION:
            return _text_input(
                asset.asset_description,
                embedding_type=embedding_type,
                source_mode=EmbeddingSourceMode.DESCRIPTION_TEXT,
            )
        return _text_input(
            _feature_text(asset.asset_features, embedding_type),
            embedding_type=embedding_type,
            source_mode=EmbeddingSourceMode.FEATURE_TEXT,
        )

    async def _native_input(self, asset: EmbeddingAsset) -> _EmbeddingInput:
        if asset.asset_type in {AssetType.MARKDOWN_BLOCK.value, AssetType.TEXT_BLOCK.value}:
            return _text_input(
                asset.raw_content,
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                source_mode=EmbeddingSourceMode.ORIGINAL_TEXT,
            )
        if asset.asset_type == AssetType.IMAGE.value:
            image_bytes = await asyncio.to_thread(_read_local_source, asset.source_storage_uri)
            image_url = _data_uri(mime_type=asset.source_mime_type, content=image_bytes)
            return _EmbeddingInput(
                input_items=[{"type": "image_url", "image_url": {"url": image_url}}],
                source_content_hash=_hash_bytes(
                    EmbeddingType.NATIVE_MULTIMODAL.value.encode("utf-8"),
                    EmbeddingSourceMode.ORIGINAL_IMAGE.value.encode("utf-8"),
                    image_bytes,
                ),
                source_mode=EmbeddingSourceMode.ORIGINAL_IMAGE,
            )
        if asset.asset_type == AssetType.VIDEO_SEGMENT.value:
            video_url = await self._video_url(asset)
            return _EmbeddingInput(
                input_items=[{"type": "video_url", "video_url": {"url": video_url}}],
                source_content_hash=_hash_text(
                    EmbeddingType.NATIVE_MULTIMODAL.value,
                    EmbeddingSourceMode.ORIGINAL_VIDEO.value,
                    asset.content_hash,
                ),
                source_mode=EmbeddingSourceMode.ORIGINAL_VIDEO,
            )
        raise EmbeddingInputUnavailable(f"unsupported native Asset type: {asset.asset_type}")

    async def _video_url(self, asset: EmbeddingAsset) -> str:
        if not asset.derived_file_uri:
            raise EmbeddingInputUnavailable("video Asset has no derived MP4")
        parsed = urlparse(asset.derived_file_uri)
        if parsed.scheme in {"http", "https"}:
            return asset.derived_file_uri
        if parsed.scheme != "s3" or self._video_url_signer is None:
            raise EmbeddingInputUnavailable(
                "video Asset needs an http(s) derived_file_uri or configured object storage signer"
            )
        signed_url = await self._video_url_signer.presigned_get_uri(asset.derived_file_uri)
        if urlparse(signed_url).scheme not in {"http", "https"}:
            raise ValueError("object storage signer did not return an http(s) URL")
        return signed_url


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


def _feature_text(features: Mapping[str, Any], embedding_type: EmbeddingType) -> str | None:
    raw = features.get(embedding_type.value)
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        value = raw.get("value")
        return value if isinstance(value, str) else None
    return None


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


def _data_uri(*, mime_type: str, content: bytes) -> str:
    if not content:
        raise EmbeddingInputUnavailable("image source file is empty")
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
