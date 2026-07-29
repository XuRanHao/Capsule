"""Transactional persistence for source files, assets, jobs, and Embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.db.base import id_factory
from capsule.db.models import Asset, EmbeddingRecord, ProcessingJob, SourceFile, Workspace
from capsule.db.session import Database
from capsule.enums import (
    AssetNameSource,
    EmbeddingStatus,
    JobStatus,
    PipelineStage,
    ProcessingStatus,
)
from capsule.schemas import AssetCreate, DiscoveredFile, StoredFileResult


class AssetRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def create_job(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        total_count: int,
    ) -> str:
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(_resolve_path(input_path)),
                total_count=total_count,
                status=JobStatus.RUNNING.value,
                current_stage=PipelineStage.PARSING.value,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.flush()
            return job.job_id

    async def get_or_create_source_file(
        self,
        *,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
    ) -> str:
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            source = await self._find_source_file(
                session,
                workspace_id=workspace_id,
                relative_path=source_file.relative_path,
                lock=True,
            )
            source_path = _resolve_path(Path(source_file.path))
            values = {
                "original_file_name": source_path.name,
                "file_type": source_file.extension,
                "mime_type": mime_type,
                "relative_path": source_file.relative_path,
                "file_tree_context": list(Path(source_file.relative_path).parent.parts)
                if Path(source_file.relative_path).parent != Path(".")
                else [],
                "storage_uri": source_path.as_uri(),
                "sha256": sha256,
                "file_size_bytes": source_file.size_bytes,
                "processing_status": ProcessingStatus.PROCESSING.value,
                "error_message": None,
            }
            if source is None:
                source = SourceFile(workspace_id=workspace_id, **values)
                session.add(source)
            else:
                for field, value in values.items():
                    setattr(source, field, value)
            await session.flush()
            return source.source_file_id

    async def replace_assets(
        self,
        *,
        source_file_id: str,
        assets: list[AssetCreate],
    ) -> StoredFileResult:
        if any(asset.source_file_id != source_file_id for asset in assets):
            raise ValueError("all assets must belong to the supplied source_file_id")

        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")

            rows = await session.scalars(
                select(Asset).where(Asset.source_file_id == source_file_id).with_for_update()
            )
            existing = {asset.asset_key: asset for asset in rows}
            stored_ids: list[str] = []

            for values in assets:
                current = existing.pop(values.asset_key, None)
                if current is None:
                    current = Asset(**_asset_values(values))
                    session.add(current)
                else:
                    content_changed = current.content_hash != values.content_hash
                    user_name = (
                        current.asset_name
                        if current.asset_name_source == AssetNameSource.USER.value
                        else None
                    )
                    _update_asset(current, values, content_changed=content_changed)
                    if user_name is not None:
                        current.asset_name = user_name
                        current.asset_name_source = AssetNameSource.USER.value
                stored_ids.append(current.asset_id)

            for stale in existing.values():
                await session.delete(stale)

            source.processing_status = ProcessingStatus.PROCESSING.value
            source.error_message = None
            await session.flush()
            return StoredFileResult(source_file_id=source_file_id, asset_ids=stored_ids)

    async def record_file_failure(
        self,
        *,
        job_id: str,
        source_file_id: str | None,
        relative_path: str,
        error: str,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.failed_count += 1
            job.error_info = [
                *job.error_info,
                {"relative_path": relative_path, "error": error[:2000]},
            ]
            if source_file_id is not None:
                source = await session.get(SourceFile, source_file_id, with_for_update=True)
                if source is not None:
                    source.processing_status = ProcessingStatus.FAILED.value
                    source.error_message = error[:2000]

    async def record_file_success(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.completed_count += 1

    async def finalize_job(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.failed_count == 0:
                job.status = JobStatus.COMPLETED.value
                job.current_stage = PipelineStage.ASSET_STORED.value
            elif job.completed_count == 0:
                job.status = JobStatus.FAILED.value
                job.current_stage = PipelineStage.FAILED.value
            else:
                job.status = JobStatus.PARTIAL_FAILED.value
                job.current_stage = PipelineStage.ASSET_STORED.value
            job.completed_at = datetime.now(UTC)

    @staticmethod
    async def _ensure_workspace(session: AsyncSession, workspace_id: str) -> None:
        if await session.get(Workspace, workspace_id) is None:
            session.add(Workspace(workspace_id=workspace_id, name=workspace_id))
            await session.flush()

    @staticmethod
    async def _find_source_file(
        session: AsyncSession,
        *,
        workspace_id: str,
        relative_path: str,
        lock: bool,
    ) -> SourceFile | None:
        statement = select(SourceFile).where(
            SourceFile.workspace_id == workspace_id,
            SourceFile.relative_path == relative_path,
        )
        if lock:
            statement = statement.with_for_update()
        return cast(SourceFile | None, await session.scalar(statement))


def _asset_values(values: AssetCreate) -> dict[str, object]:
    data = values.model_dump(mode="json")
    data["asset_type"] = values.asset_type.value
    data["processing_status"] = values.processing_status.value
    data["source_contexts"] = [context.model_dump() for context in values.source_contexts]
    return data


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _update_asset(current: Asset, values: AssetCreate, *, content_changed: bool) -> None:
    current.workspace_id = values.workspace_id
    current.asset_type = values.asset_type.value
    current.file_name = values.file_name
    current.file_type = values.file_type
    current.content_hash = values.content_hash
    current.file_tree_context = values.file_tree_context
    current.source_contexts = [context.model_dump() for context in values.source_contexts]
    current.file_info = values.file_info
    current.source_locator = values.source_locator
    current.raw_content = values.raw_content
    current.derived_file_uri = values.derived_file_uri
    current.preview_uri = values.preview_uri
    current.processing_status = ProcessingStatus.PENDING.value
    current.error_message = None
    if content_changed:
        if current.asset_name_source != AssetNameSource.USER.value:
            current.asset_name = None
            current.asset_name_source = None
        current.asset_description = None
        current.asset_features = {}
        current.feature_revision += 1
        current.embedding_revision += 1


#===========================================
#      Embedding persistence
#===========================================


@dataclass(slots=True, frozen=True)
class EmbeddingAsset:
    """The Asset fields needed to construct one model Embedding input."""

    asset_id: str
    workspace_id: str
    project_id: str
    source_file_id: str
    asset_type: str
    file_type: str
    content_hash: str
    embedding_revision: int
    created_at: datetime
    raw_content: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    derived_file_uri: str | None
    source_storage_uri: str
    source_mime_type: str


@dataclass(slots=True, frozen=True)
class PreparedEmbedding:
    """A durable record reserved before calling the model or Milvus."""

    embedding_id: str
    milvus_primary_key: str
    already_indexed: bool


class EmbeddingRepository:
    """Own PostgreSQL metadata and state transitions for vector persistence."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._new_embedding_id = id_factory("emb")

    async def list_assets(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str] | None = None,
    ) -> list[EmbeddingAsset]:
        statement = (
            select(Asset, SourceFile)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(Asset.workspace_id == workspace_id)
            .order_by(Asset.created_at, Asset.asset_id)
        )
        if asset_ids:
            statement = statement.where(Asset.asset_id.in_(asset_ids))

        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [
            EmbeddingAsset(
                asset_id=asset.asset_id,
                workspace_id=asset.workspace_id,
                project_id=asset.project_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                file_type=asset.file_type,
                content_hash=asset.content_hash,
                embedding_revision=asset.embedding_revision,
                created_at=asset.created_at,
                raw_content=asset.raw_content,
                asset_description=asset.asset_description,
                asset_features=dict(asset.asset_features),
                derived_file_uri=asset.derived_file_uri,
                source_storage_uri=source.storage_uri,
                source_mime_type=source.mime_type,
            )
            for asset, source in rows
        ]

    async def prepare(
        self,
        *,
        asset: EmbeddingAsset,
        embedding_type: str,
        model_name: str,
        dimension: int,
        source_content_hash: str,
        source_mode: str,
        milvus_collection: str,
        force: bool,
    ) -> PreparedEmbedding:
        """Reserve one stable primary key or return an existing indexed record."""
        async with self._database.session() as session, session.begin():
            statement = (
                select(EmbeddingRecord)
                .where(
                    EmbeddingRecord.asset_id == asset.asset_id,
                    EmbeddingRecord.embedding_type == embedding_type,
                    EmbeddingRecord.model_name == model_name,
                    EmbeddingRecord.dimension == dimension,
                    EmbeddingRecord.source_content_hash == source_content_hash,
                )
                .with_for_update()
            )
            record = await session.scalar(statement)
            if record is not None:
                if record.status == EmbeddingStatus.INDEXED.value and not force:
                    return PreparedEmbedding(
                        embedding_id=record.embedding_id,
                        milvus_primary_key=record.milvus_primary_key,
                        already_indexed=True,
                    )
                record.status = EmbeddingStatus.PROCESSING.value
                record.latency_ms = None
                record.usage = {}
                return PreparedEmbedding(
                    embedding_id=record.embedding_id,
                    milvus_primary_key=record.milvus_primary_key,
                    already_indexed=False,
                )

            embedding_id = self._new_embedding_id()
            record = EmbeddingRecord(
                embedding_id=embedding_id,
                workspace_id=asset.workspace_id,
                project_id=asset.project_id,
                asset_id=asset.asset_id,
                embedding_type=embedding_type,
                model_name=model_name,
                dimension=dimension,
                source_content_hash=source_content_hash,
                embedding_source_mode=source_mode,
                milvus_collection=milvus_collection,
                milvus_primary_key=embedding_id,
                status=EmbeddingStatus.PROCESSING.value,
            )
            session.add(record)
            return PreparedEmbedding(
                embedding_id=embedding_id,
                milvus_primary_key=embedding_id,
                already_indexed=False,
            )

    async def mark_indexed(
        self,
        *,
        embedding_id: str,
        latency_ms: int,
        usage: dict[str, Any],
    ) -> None:
        async with self._database.session() as session, session.begin():
            record = await session.get(EmbeddingRecord, embedding_id, with_for_update=True)
            if record is None:
                raise ValueError(f"embedding record does not exist: {embedding_id}")
            record.status = EmbeddingStatus.INDEXED.value
            record.latency_ms = latency_ms
            record.usage = usage

    async def mark_failed(self, *, embedding_id: str) -> None:
        async with self._database.session() as session, session.begin():
            record = await session.get(EmbeddingRecord, embedding_id, with_for_update=True)
            if record is not None:
                record.status = EmbeddingStatus.FAILED.value
