"""Transactional persistence for source files, assets, jobs, and Embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.db.base import id_factory
from capsule.db.models import (
    Asset,
    ClusterCapsule,
    ClusterMembership,
    ClusterRepresentativeAsset,
    ClusterRun,
    EmbeddingRecord,
    ModelCallLog,
    ProcessingJob,
    SourceFile,
    Workspace,
)
from capsule.db.session import Database
from capsule.enums import (
    AssetNameSource,
    AssetType,
    ClusterInternalVariance,
    ClusterRunStatus,
    EmbeddingStatus,
    JobStatus,
    PipelineStage,
    ProcessingStatus,
)
from capsule.features import embedding_channel_is_eligible
from capsule.schemas import (
    AssetCreate,
    AssetEmbeddingState,
    AssetListResponse,
    AssetSourceRecord,
    AssetUnderstanding,
    AssetViewRecord,
    ClusterCapsuleRecord,
    ClusterCapsuleWrite,
    ClusterMemberRecord,
    ClusterRepresentativeWrite,
    ClusterRunRecord,
    DiscoveredFile,
    ProcessingJobRecord,
    StoredFileResult,
)


@dataclass(slots=True, frozen=True)
class AssetMediaTarget:
    asset_id: str
    workspace_id: str
    asset_type: str
    source_storage_uri: str
    source_mime_type: str
    preview_uri: str | None
    derived_file_uri: str | None


@dataclass(slots=True, frozen=True)
class PreparedSourceFile:
    source_file_id: str
    already_processed: bool
    asset_count: int = 0
    generation: int = 0


class StaleAssetGenerationError(ValueError):
    """A delayed queue delivery belongs to an older source processing run."""


class LibraryClearBusyError(ValueError):
    """The asset library cannot be cleared while an import is still active."""


@dataclass(slots=True, frozen=True)
class LibraryClearSnapshot:
    """Durable records collected before every workspace is deleted."""

    workspace_count: int
    asset_count: int
    source_file_count: int
    embedding_count: int
    job_count: int


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

    async def create_pending_import_job(
        self,
        *,
        workspace_id: str,
        import_root: Path,
    ) -> str:
        """Create the durable upload session before browser files are transferred."""
        root = _resolve_path(import_root)
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(root),
                total_count=0,
                status=JobStatus.QUEUED.value,
                current_stage=PipelineStage.DISCOVERING.value,
            )
            session.add(job)
            await session.flush()
            job.input_path = str(root / job.job_id)
            return job.job_id

    async def clear_all_records(self) -> LibraryClearSnapshot:
        """Clear every workspace and its owned records from PostgreSQL.

        PostgreSQL cascades remove all workspace-owned assets, jobs, Embeddings,
        Cluster runs, search history, and query-image metadata. Orphaned model
        call logs are deleted explicitly. External cleanup targets are returned
        for the caller to remove after this transaction.
        """
        active_statuses = (
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.RETRYING.value,
        )
        async with self._database.session() as session, session.begin():
            workspaces = list(
                await session.scalars(select(Workspace).with_for_update())
            )

            active_job_count = int(
                await session.scalar(
                    select(func.count(ProcessingJob.job_id)).where(
                        ProcessingJob.status.in_(active_statuses)
                    )
                )
                or 0
            )
            if active_job_count:
                raise LibraryClearBusyError(
                    f"asset library has {active_job_count} active import job(s)"
                )

            asset_count = int(
                await session.scalar(select(func.count(Asset.asset_id))) or 0
            )
            source_file_count = int(
                await session.scalar(select(func.count(SourceFile.source_file_id))) or 0
            )
            job_count = int(
                await session.scalar(select(func.count(ProcessingJob.job_id))) or 0
            )
            embedding_count = int(
                await session.scalar(select(func.count(EmbeddingRecord.embedding_id)))
                or 0
            )
            await session.execute(delete(ModelCallLog))
            for workspace in workspaces:
                await session.delete(workspace)

            return LibraryClearSnapshot(
                workspace_count=len(workspaces),
                asset_count=asset_count,
                source_file_count=source_file_count,
                embedding_count=embedding_count,
                job_count=job_count,
            )

    async def start_import_job(self, *, job_id: str, total_count: int) -> None:
        """Freeze an upload session and make it available to ``PipelineRunner``."""
        if total_count < 1:
            raise ValueError("an import job must contain at least one file")
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status != JobStatus.QUEUED.value:
                raise ValueError(f"processing job cannot be started from status {job.status}")
            job.total_count = total_count
            job.status = JobStatus.RUNNING.value
            job.current_stage = PipelineStage.PARSING.value
            job.started_at = datetime.now(UTC)

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

    async def prepare_source_file(
        self,
        *,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
    ) -> "PreparedSourceFile":
        """Claim a logical source or reuse its completed, byte-identical assets."""
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            source = await self._find_source_file(
                session,
                workspace_id=workspace_id,
                relative_path=source_file.relative_path,
                lock=True,
            )
            source_path = _resolve_path(Path(source_file.path))
            metadata = {
                "original_file_name": source_path.name,
                "file_type": source_file.extension,
                "mime_type": mime_type,
                "relative_path": source_file.relative_path,
                "file_tree_context": list(Path(source_file.relative_path).parent.parts)
                if Path(source_file.relative_path).parent != Path(".")
                else [],
                "storage_uri": source_path.as_uri(),
                "file_size_bytes": source_file.size_bytes,
            }
            if (
                source is not None
                and source.sha256 == sha256
                and source.processing_fingerprint == processing_fingerprint
                and source.processing_status == ProcessingStatus.COMPLETED.value
            ):
                for field, value in metadata.items():
                    setattr(source, field, value)
                asset_count = int(
                    await session.scalar(
                        select(func.count(Asset.asset_id)).where(
                            Asset.source_file_id == source.source_file_id
                        )
                    )
                    or 0
                )
                return PreparedSourceFile(
                    source_file_id=source.source_file_id,
                    already_processed=True,
                    asset_count=asset_count,
                    generation=source.processing_generation,
                )

            values = {
                **metadata,
                "sha256": sha256,
                "processing_fingerprint": processing_fingerprint,
                "processing_status": ProcessingStatus.PROCESSING.value,
                "error_message": None,
            }
            if source is None:
                source = SourceFile(
                    workspace_id=workspace_id,
                    processing_generation=1,
                    **values,
                )
                session.add(source)
            else:
                for field, value in values.items():
                    setattr(source, field, value)
                source.processing_generation += 1
            await session.flush()
            return PreparedSourceFile(
                source_file_id=source.source_file_id,
                already_processed=False,
                generation=source.processing_generation,
            )

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
            generation = source.processing_generation
            if any(asset.generation != generation for asset in assets):
                raise ValueError("asset generation does not match the source generation")

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

            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            await session.flush()
            return StoredFileResult(source_file_id=source_file_id, asset_ids=stored_ids)

    async def upsert_generated_asset(
        self,
        *,
        source_file_id: str,
        generation: int,
        asset: AssetCreate,
    ) -> str:
        """Store one completed Segment, rejecting deliveries from an obsolete run."""
        if asset.source_file_id != source_file_id or asset.generation != generation:
            raise ValueError("asset does not belong to the supplied source generation")
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            current = await session.scalar(
                select(Asset)
                .where(
                    Asset.source_file_id == source_file_id,
                    Asset.asset_key == asset.asset_key,
                )
                .with_for_update()
            )
            if current is None:
                current = Asset(**_asset_values(asset))
                session.add(current)
            else:
                content_changed = current.content_hash != asset.content_hash
                user_name = (
                    current.asset_name
                    if current.asset_name_source == AssetNameSource.USER.value
                    else None
                )
                _update_asset(current, asset, content_changed=content_changed)
                if user_name is not None:
                    current.asset_name = user_name
                    current.asset_name_source = AssetNameSource.USER.value
            await session.flush()
            return current.asset_id

    async def assert_current_generation(
        self,
        *,
        source_file_id: str,
        generation: int,
    ) -> None:
        """Reject stale queue work before it can overwrite a stable object key."""
        async with self._database.session() as session:
            current_generation = await session.scalar(
                select(SourceFile.processing_generation).where(
                    SourceFile.source_file_id == source_file_id
                )
            )
        if current_generation is None:
            raise ValueError(f"source file does not exist: {source_file_id}")
        if current_generation != generation:
            raise StaleAssetGenerationError(
                f"source generation advanced from {generation} to {current_generation}"
            )

    async def finalize_asset_generation(
        self,
        *,
        source_file_id: str,
        generation: int,
    ) -> StoredFileResult:
        """Publish one complete generation and remove stale Asset rows atomically."""
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            await session.execute(
                delete(Asset).where(
                    Asset.source_file_id == source_file_id,
                    Asset.generation != generation,
                )
            )
            asset_ids = list(
                await session.scalars(
                    select(Asset.asset_id)
                    .where(
                        Asset.source_file_id == source_file_id,
                        Asset.generation == generation,
                    )
                    .order_by(Asset.asset_key)
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            return StoredFileResult(source_file_id=source_file_id, asset_ids=asset_ids)

    async def finalize_asset_generation_if_complete(
        self,
        *,
        source_file_id: str,
        generation: int,
        expected_asset_count: int,
    ) -> bool:
        """Finalize a recovered stream generation once every Segment is durable."""
        if expected_asset_count < 1:
            raise ValueError("expected_asset_count must be positive")
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            stored_count = int(
                await session.scalar(
                    select(func.count(Asset.asset_id)).where(
                        Asset.source_file_id == source_file_id,
                        Asset.generation == generation,
                    )
                )
                or 0
            )
            if stored_count < expected_asset_count:
                return False
            if stored_count > expected_asset_count:
                raise ValueError("source generation contains more Assets than its upload manifest")
            await session.execute(
                delete(Asset).where(
                    Asset.source_file_id == source_file_id,
                    Asset.generation != generation,
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            return True

    async def record_file_failure(
        self,
        *,
        job_id: str,
        source_file_id: str | None,
        relative_path: str,
        error: str,
        generation: int | None = None,
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
                if source is not None and (
                    generation is None or source.processing_generation == generation
                ):
                    source.processing_status = ProcessingStatus.FAILED.value
                    source.error_message = error[:2000]

    async def record_file_success(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.completed_count += 1

    async def add_job_stage_durations(
        self,
        *,
        job_id: str,
        durations_ms: dict[str, float],
    ) -> None:
        """Accumulate measured wall-clock work for independently timed stages."""
        if not durations_ms:
            return
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            current = dict(job.stage_durations_ms)
            for stage, duration_ms in durations_ms.items():
                current[stage] = round(
                    current.get(stage, 0.0) + max(0.0, duration_ms),
                    3,
                )
            job.stage_durations_ms = current

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

    async def fail_job(self, *, job_id: str, error: str) -> None:
        """Fail an import before per-file handling could record an outcome."""
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.status = JobStatus.FAILED.value
            job.current_stage = PipelineStage.FAILED.value
            job.error_info = [*job.error_info, {"relative_path": "", "error": error[:2000]}]
            if job.failed_count == 0:
                job.failed_count = 1
            job.completed_at = datetime.now(UTC)

    async def get_job(self, *, job_id: str, workspace_id: str) -> ProcessingJobRecord:
        async with self._database.session() as session:
            job = await session.scalar(
                select(ProcessingJob).where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            return ProcessingJobRecord(
                job_id=job.job_id,
                workspace_id=job.workspace_id,
                input_path=job.input_path,
                total_count=job.total_count,
                completed_count=job.completed_count,
                failed_count=job.failed_count,
                status=job.status,
                current_stage=job.current_stage,
                error_info=list(job.error_info),
                stage_durations_ms=dict(job.stage_durations_ms),
                started_at=job.started_at,
                completed_at=job.completed_at,
            )

    async def list_jobs(
        self,
        *,
        workspace_id: str,
        limit: int = 50,
    ) -> list[ProcessingJobRecord]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(ProcessingJob)
                .where(ProcessingJob.workspace_id == workspace_id)
                .order_by(ProcessingJob.created_at.desc(), ProcessingJob.job_id.desc())
                .limit(limit)
            )
            return [_processing_job_record(job) for job in rows]

    async def list_asset_views(
        self,
        *,
        workspace_id: str,
        asset_type: str | None = None,
        processing_status: str | None = None,
        source_file_id: str | None = None,
        query: str | None = None,
        asset_ids: Sequence[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AssetListResponse:
        filters = [
            Asset.workspace_id == workspace_id,
            Asset.generation == SourceFile.processing_generation,
        ]
        if asset_type:
            filters.append(Asset.asset_type == asset_type)
        if processing_status:
            filters.append(Asset.processing_status == processing_status)
        if source_file_id:
            filters.append(Asset.source_file_id == source_file_id)
        if asset_ids:
            filters.append(Asset.asset_id.in_(asset_ids))
        normalized_query = query.strip() if query else ""
        if normalized_query:
            pattern = f"%{normalized_query}%"
            filters.append(
                or_(
                    Asset.file_name.ilike(pattern),
                    Asset.asset_name.ilike(pattern),
                    Asset.asset_description.ilike(pattern),
                    SourceFile.relative_path.ilike(pattern),
                )
            )

        async with self._database.session() as session:
            total = int(
                await session.scalar(
                    select(func.count())
                    .select_from(Asset)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(*filters)
                )
                or 0
            )
            rows = (
                await session.execute(
                    select(Asset, SourceFile)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(*filters)
                    .order_by(Asset.created_at.desc(), Asset.asset_id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
            embeddings = await _latest_embedding_states(
                session,
                [asset.asset_id for asset, _ in rows],
            )
        return AssetListResponse(
            items=[
                _asset_view_record(
                    asset,
                    source,
                    embeddings=embeddings.get(asset.asset_id, []),
                )
                for asset, source in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_asset_view(
        self,
        *,
        asset_id: str,
        workspace_id: str,
    ) -> AssetViewRecord:
        result = await self.list_asset_views(
            workspace_id=workspace_id,
            asset_ids=[asset_id],
            limit=1,
        )
        if not result.items:
            raise ValueError(f"asset does not exist: {asset_id}")
        return result.items[0]

    async def get_asset_media(
        self,
        *,
        asset_id: str,
        workspace_id: str,
    ) -> AssetMediaTarget:
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(Asset, SourceFile)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(
                        Asset.asset_id == asset_id,
                        Asset.workspace_id == workspace_id,
                        Asset.generation == SourceFile.processing_generation,
                    )
                )
            ).one_or_none()
        if row is None:
            raise ValueError(f"asset does not exist: {asset_id}")
        asset, source = row
        return AssetMediaTarget(
            asset_id=asset.asset_id,
            workspace_id=asset.workspace_id,
            asset_type=asset.asset_type,
            source_storage_uri=source.storage_uri,
            source_mime_type=source.mime_type,
            preview_uri=asset.preview_uri,
            derived_file_uri=asset.derived_file_uri,
        )

    async def set_job_stage(
        self,
        *,
        job_id: str,
        stage: PipelineStage,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.status = JobStatus.RUNNING.value
            job.current_stage = stage.value
            job.completed_at = None

    async def begin_asset_enrichment(self, *, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        async with self._database.session() as session, session.begin():
            rows = await session.scalars(
                select(Asset).where(Asset.asset_id.in_(asset_ids)).with_for_update()
            )
            for asset in rows:
                asset.processing_status = ProcessingStatus.PROCESSING.value
                asset.error_message = None

    async def store_understanding(
        self,
        *,
        asset_id: str,
        understanding: AssetUnderstanding,
    ) -> None:
        async with self._database.session() as session, session.begin():
            asset = await session.get(Asset, asset_id, with_for_update=True)
            if asset is None:
                raise ValueError(f"asset does not exist: {asset_id}")
            features = understanding.features.model_dump(mode="json")
            semantic_changed = asset.asset_description not in {
                None,
                understanding.asset_description,
            } or (bool(asset.asset_features) and asset.asset_features != features)
            if asset.asset_name_source != AssetNameSource.USER.value:
                asset.asset_name = understanding.asset_name
                asset.asset_name_source = AssetNameSource.MODEL.value
            asset.asset_description = understanding.asset_description
            asset.asset_features = features
            asset.processing_status = ProcessingStatus.PROCESSING.value
            asset.error_message = None
            if semantic_changed:
                asset.feature_revision += 1
                asset.embedding_revision += 1

    async def finalize_enrichment(
        self,
        *,
        job_id: str,
        asset_ids: Sequence[str],
        errors: Sequence[dict[str, str]],
    ) -> None:
        errors_by_asset: dict[str, list[str]] = {}
        for error in errors:
            asset_id = error.get("asset_id", "")
            if asset_id:
                errors_by_asset.setdefault(asset_id, []).append(error.get("error", "failed"))

        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            rows = list(
                await session.scalars(
                    select(Asset).where(Asset.asset_id.in_(asset_ids)).with_for_update()
                )
            )
            for asset in rows:
                asset_errors = errors_by_asset.get(asset.asset_id, [])
                if asset_errors:
                    asset.processing_status = ProcessingStatus.PARTIAL_FAILED.value
                    asset.error_message = "; ".join(asset_errors)[:2000]
                else:
                    asset.processing_status = ProcessingStatus.COMPLETED.value
                    asset.error_message = None

            source_ids = {asset.source_file_id for asset in rows}
            for source_id in source_ids:
                source = await session.get(SourceFile, source_id, with_for_update=True)
                if source is None:
                    continue
                # Incremental video Assets can finish Understanding while later
                # Segments are still rendering/uploading. Only generation
                # finalization may publish the SourceFile in that window.
                if source.processing_status == ProcessingStatus.PROCESSING.value:
                    continue
                source_assets = list(
                    await session.scalars(
                        select(Asset).where(
                            Asset.source_file_id == source_id,
                            Asset.generation == source.processing_generation,
                        )
                    )
                )
                failed_assets = [
                    asset
                    for asset in source_assets
                    if asset.processing_status
                    in {
                        ProcessingStatus.FAILED.value,
                        ProcessingStatus.PARTIAL_FAILED.value,
                    }
                ]
                source.processing_status = (
                    ProcessingStatus.PARTIAL_FAILED.value
                    if failed_assets
                    else ProcessingStatus.COMPLETED.value
                )
                source.error_message = (
                    "; ".join(
                        asset.error_message or "asset enrichment failed" for asset in failed_assets
                    )[:2000]
                    if failed_assets
                    else None
                )

            if errors:
                job.status = JobStatus.PARTIAL_FAILED.value
                job.error_info = [
                    *job.error_info,
                    *[
                        {
                            "asset_id": error.get("asset_id", ""),
                            "stage": error.get("stage", ""),
                            "error": error.get("error", "")[:2000],
                        }
                        for error in errors
                    ],
                ]
            elif job.failed_count:
                job.status = JobStatus.PARTIAL_FAILED.value
            else:
                job.status = JobStatus.COMPLETED.value
            job.current_stage = PipelineStage.COMPLETED.value
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


def _processing_job_record(job: ProcessingJob) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id=job.job_id,
        workspace_id=job.workspace_id,
        input_path=job.input_path,
        total_count=job.total_count,
        completed_count=job.completed_count,
        failed_count=job.failed_count,
        status=job.status,
        current_stage=job.current_stage,
        error_info=list(job.error_info),
        stage_durations_ms=dict(job.stage_durations_ms),
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


async def _latest_embedding_states(
    session: AsyncSession,
    asset_ids: list[str],
) -> dict[str, list[AssetEmbeddingState]]:
    if not asset_ids:
        return {}
    rows = await session.scalars(
        select(EmbeddingRecord)
        .where(EmbeddingRecord.asset_id.in_(asset_ids))
        .order_by(
            EmbeddingRecord.asset_id,
            EmbeddingRecord.embedding_type,
            EmbeddingRecord.created_at.desc(),
            EmbeddingRecord.embedding_id.desc(),
        )
    )
    selected: dict[tuple[str, str], AssetEmbeddingState] = {}
    for record in rows:
        key = (record.asset_id, record.embedding_type)
        if key in selected:
            continue
        selected[key] = AssetEmbeddingState(
            embedding_type=record.embedding_type,
            status=record.status,
            model_name=record.model_name,
        )
    grouped: dict[str, list[AssetEmbeddingState]] = {}
    for (asset_id, _), state in selected.items():
        grouped.setdefault(asset_id, []).append(state)
    return grouped


def _asset_view_record(
    asset: Asset,
    source: SourceFile,
    *,
    embeddings: list[AssetEmbeddingState],
) -> AssetViewRecord:
    return AssetViewRecord(
        asset_id=asset.asset_id,
        workspace_id=asset.workspace_id,
        project_id=asset.project_id,
        source_file_id=asset.source_file_id,
        asset_type=AssetType(asset.asset_type),
        file_name=asset.file_name,
        file_type=asset.file_type,
        asset_name=asset.asset_name,
        asset_description=asset.asset_description,
        asset_features=dict(asset.asset_features),
        file_tree_context=list(asset.file_tree_context),
        source_contexts=list(asset.source_contexts),
        file_info=dict(asset.file_info),
        source_locator=dict(asset.source_locator),
        raw_content=asset.raw_content,
        processing_status=asset.processing_status,
        feature_revision=asset.feature_revision,
        embedding_revision=asset.embedding_revision,
        error_message=asset.error_message,
        source_file=AssetSourceRecord(
            source_file_id=source.source_file_id,
            original_file_name=source.original_file_name,
            relative_path=source.relative_path,
            file_type=source.file_type,
            mime_type=source.mime_type,
            file_size_bytes=source.file_size_bytes,
            processing_status=source.processing_status,
            error_message=source.error_message,
        ),
        embeddings=embeddings,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
    )


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
    current.generation = values.generation
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


# ===========================================
#      Embedding persistence
# ===========================================


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
    file_name: str = ""
    source_relative_path: str = ""
    file_tree_context: list[str] = field(default_factory=list)
    source_contexts: list[dict[str, Any]] = field(default_factory=list)
    file_info: dict[str, Any] = field(default_factory=dict)
    source_locator: dict[str, Any] = field(default_factory=dict)


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
            .where(
                Asset.workspace_id == workspace_id,
                Asset.generation == SourceFile.processing_generation,
            )
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
                file_name=asset.file_name,
                source_relative_path=source.relative_path,
                file_tree_context=list(asset.file_tree_context),
                source_contexts=list(asset.source_contexts),
                file_info=dict(asset.file_info),
                source_locator=dict(asset.source_locator),
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

    async def list_indexed_cluster_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> list["ClusterEmbeddingAsset"]:
        """Return the latest indexed vector record for each Asset in one channel."""
        statement = (
            select(EmbeddingRecord, Asset, SourceFile.relative_path)
            .join(Asset, Asset.asset_id == EmbeddingRecord.asset_id)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                EmbeddingRecord.workspace_id == workspace_id,
                EmbeddingRecord.embedding_type == embedding_type,
                EmbeddingRecord.model_name == model_name,
                EmbeddingRecord.dimension == dimension,
                EmbeddingRecord.milvus_collection == milvus_collection,
                EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
                Asset.generation == SourceFile.processing_generation,
            )
            .order_by(EmbeddingRecord.created_at.desc(), EmbeddingRecord.embedding_id.desc())
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        # A regenerated Asset can have historical indexed records.  Keep its
        # newest record so one Asset contributes at most one vector per type.
        selected: dict[str, ClusterEmbeddingAsset] = {}
        for record, asset, source_relative_path in rows:
            if asset.asset_id in selected:
                continue
            if not embedding_channel_is_eligible(
                embedding_type=embedding_type,
                asset_features=asset.asset_features,
                asset_description=asset.asset_description,
            ):
                continue
            selected[asset.asset_id] = ClusterEmbeddingAsset(
                embedding_id=record.embedding_id,
                asset_id=asset.asset_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                asset_name=asset.asset_name,
                asset_description=asset.asset_description,
                asset_features=dict(asset.asset_features),
                file_tree_context=list(asset.file_tree_context),
                source_relative_path=source_relative_path,
            )
        return list(selected.values())


# ===========================================
#      Cluster Capsule persistence
# ===========================================


class ClusterRepository:
    """Persist model summaries, representative Asset IDs, and user overrides."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def create_pending_run(self, *, workspace_id: str, embedding_type: str) -> str:
        """Reserve a ClusterRun ID for an HTTP request before background execution."""
        async with self._database.session() as session, session.begin():
            if await session.get(Workspace, workspace_id) is None:
                raise ValueError(f"workspace does not exist: {workspace_id}")
            run = ClusterRun(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                input_embedding_ids=[],
                dataset_hash="0" * 64,
                sample_count=0,
                preprocessing={"submission_mode": "async_api"},
                parameters={},
                status=ClusterRunStatus.PENDING.value,
            )
            session.add(run)
            await session.flush()
            return run.cluster_run_id

    async def start_pending_run(
        self,
        *,
        cluster_run_id: str,
        workspace_id: str,
        embedding_type: str,
        embedding_ids: list[str],
        dataset_hash: str,
        preprocessing: dict[str, Any],
        parameters: dict[str, Any],
    ) -> None:
        """Populate a pending API run once its exact vector inputs are loaded."""
        async with self._database.session() as session, session.begin():
            run = await _load_run_for_update(session, cluster_run_id)
            if run.workspace_id != workspace_id or run.embedding_type != embedding_type:
                raise ValueError("pending run workspace or embedding type does not match")
            if run.status != ClusterRunStatus.PENDING.value:
                raise ValueError(f"cluster run is not pending: {cluster_run_id}")
            run.input_embedding_ids = embedding_ids
            run.dataset_hash = dataset_hash
            run.sample_count = len(embedding_ids)
            run.preprocessing = preprocessing
            run.parameters = parameters
            run.status = ClusterRunStatus.RUNNING.value
            run.started_at = datetime.now(UTC)

    async def create_run(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        embedding_ids: list[str],
        dataset_hash: str,
        preprocessing: dict[str, Any],
        parameters: dict[str, Any],
    ) -> str:
        """Create an independently auditable run for exactly one embedding type."""
        async with self._database.session() as session, session.begin():
            if await session.get(Workspace, workspace_id) is None:
                raise ValueError(f"workspace does not exist: {workspace_id}")
            run = ClusterRun(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                input_embedding_ids=embedding_ids,
                dataset_hash=dataset_hash,
                sample_count=len(embedding_ids),
                preprocessing=preprocessing,
                parameters=parameters,
                status=ClusterRunStatus.RUNNING.value,
                started_at=datetime.now(UTC),
            )
            session.add(run)
            await session.flush()
            return run.cluster_run_id

    async def complete_run(
        self,
        *,
        cluster_run_id: str,
        cluster_count: int,
        noise_count: int,
        noise_ratio: float,
        status: ClusterRunStatus = ClusterRunStatus.COMPLETED,
        preprocessing: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        async with self._database.session() as session, session.begin():
            run = await _load_run_for_update(session, cluster_run_id)
            run.cluster_count = cluster_count
            run.noise_count = noise_count
            run.noise_ratio = noise_ratio
            run.status = status.value
            if preprocessing is not None:
                run.preprocessing = preprocessing
            if parameters is not None:
                run.parameters = parameters
            run.completed_at = datetime.now(UTC)

    async def fail_run(self, *, cluster_run_id: str, error: str) -> None:
        """Mark a run failed while retaining its inputs and execution context."""
        async with self._database.session() as session, session.begin():
            run = await _load_run_for_update(session, cluster_run_id)
            run.status = ClusterRunStatus.FAILED.value
            run.preprocessing = {**run.preprocessing, "error": error[:2000]}
            run.completed_at = datetime.now(UTC)

    async def store_memberships(
        self,
        *,
        cluster_run_id: str,
        memberships: list["ClusterMembershipWrite"],
    ) -> None:
        """Store all members, including HDBSCAN noise, for one immutable run."""
        async with self._database.session() as session, session.begin():
            await _load_run_for_update(session, cluster_run_id)
            session.add_all(
                [
                    ClusterMembership(
                        cluster_run_id=cluster_run_id,
                        cluster_capsule_id=item.cluster_capsule_id,
                        asset_id=item.asset_id,
                        hdbscan_label=item.hdbscan_label,
                        membership_probability=item.membership_probability,
                        is_noise=item.is_noise,
                        distance_to_representative=item.distance_to_representative,
                    )
                    for item in memberships
                ]
            )

    async def get_run(
        self,
        *,
        cluster_run_id: str,
        workspace_id: str,
    ) -> ClusterRunRecord:
        async with self._database.session() as session:
            run = await session.scalar(
                select(ClusterRun).where(
                    ClusterRun.cluster_run_id == cluster_run_id,
                    ClusterRun.workspace_id == workspace_id,
                )
            )
            if run is None:
                raise ValueError(f"cluster run does not exist: {cluster_run_id}")
            return _cluster_run_record(run)

    async def list_runs(
        self,
        *,
        workspace_id: str,
        limit: int = 50,
    ) -> list[ClusterRunRecord]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(ClusterRun)
                .where(ClusterRun.workspace_id == workspace_id)
                .order_by(ClusterRun.started_at.desc(), ClusterRun.cluster_run_id.desc())
                .limit(limit)
            )
            return [_cluster_run_record(run) for run in rows]

    async def list_capsules(
        self,
        *,
        cluster_run_id: str,
        workspace_id: str,
    ) -> list[ClusterCapsuleRecord]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(ClusterCapsule)
                .where(
                    ClusterCapsule.cluster_run_id == cluster_run_id,
                    ClusterCapsule.workspace_id == workspace_id,
                )
                .order_by(ClusterCapsule.cluster_label)
            )
            return [_cluster_capsule_record(capsule) for capsule in rows]

    async def list_members(
        self,
        *,
        cluster_capsule_id: str,
        workspace_id: str,
    ) -> list[ClusterMemberRecord]:
        async with self._database.session() as session:
            capsule = await session.scalar(
                select(ClusterCapsule).where(
                    ClusterCapsule.cluster_capsule_id == cluster_capsule_id,
                    ClusterCapsule.workspace_id == workspace_id,
                )
            )
            if capsule is None:
                raise ValueError(f"cluster capsule does not exist: {cluster_capsule_id}")
            rows = (
                await session.execute(
                    select(ClusterMembership, Asset, SourceFile)
                    .join(Asset, Asset.asset_id == ClusterMembership.asset_id)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(
                        ClusterMembership.cluster_capsule_id == cluster_capsule_id,
                        Asset.workspace_id == workspace_id,
                        Asset.generation == SourceFile.processing_generation,
                    )
                    .order_by(
                        ClusterMembership.membership_probability.desc(),
                        Asset.asset_id,
                    )
                )
            ).all()
            return [
                ClusterMemberRecord(
                    asset_id=asset.asset_id,
                    asset_type=asset.asset_type,
                    file_name=asset.file_name,
                    asset_name=asset.asset_name,
                    asset_description=asset.asset_description,
                    source_file_id=asset.source_file_id,
                    relative_path=source.relative_path,
                    hdbscan_label=membership.hdbscan_label,
                    membership_probability=membership.membership_probability,
                    is_noise=membership.is_noise,
                    distance_to_representative=membership.distance_to_representative,
                )
                for membership, asset, source in rows
            ]

    async def upsert_capsule(self, values: ClusterCapsuleWrite) -> ClusterCapsuleRecord:
        """Store a generated summary without replacing existing user overrides."""
        _validate_representatives(values.representatives)
        representative_ids = [item.asset_id for item in values.representatives]
        medoid = next(item for item in values.representatives if item.role.value == "medoid")

        async with self._database.session() as session, session.begin():
            run = await session.get(ClusterRun, values.cluster_run_id, with_for_update=True)
            if run is None:
                raise ValueError(f"cluster run does not exist: {values.cluster_run_id}")
            if run.workspace_id != values.workspace_id:
                raise ValueError("cluster run belongs to another workspace")
            if run.embedding_type != values.embedding_type:
                raise ValueError("cluster run embedding type does not match capsule")

            assets = await _load_representative_assets(
                session,
                workspace_id=values.workspace_id,
                representative_ids=representative_ids,
            )
            _validate_representative_source_limits(values.representatives, assets)

            capsule = await session.scalar(
                select(ClusterCapsule)
                .where(
                    ClusterCapsule.cluster_run_id == values.cluster_run_id,
                    ClusterCapsule.cluster_label == values.cluster_label,
                )
                .with_for_update()
            )
            if capsule is None:
                capsule = ClusterCapsule(
                    cluster_run_id=values.cluster_run_id,
                    workspace_id=values.workspace_id,
                    embedding_type=values.embedding_type,
                    cluster_label=values.cluster_label,
                    model_generated_name=values.summary.name,
                    effective_name=values.summary.name,
                    model_generated_description=values.summary.description,
                    effective_description=values.summary.description,
                    keywords=values.summary.keywords,
                    common_features=values.summary.common_features,
                    internal_variance=values.summary.internal_variance.value,
                    member_count=values.member_count,
                    average_membership_probability=values.average_membership_probability,
                    medoid_asset_id=medoid.asset_id,
                    representative_asset_ids=representative_ids,
                )
                session.add(capsule)
                await session.flush()
            else:
                capsule.embedding_type = values.embedding_type
                capsule.model_generated_name = values.summary.name
                capsule.effective_name = capsule.user_override_name or values.summary.name
                capsule.model_generated_description = values.summary.description
                capsule.effective_description = (
                    capsule.user_override_description or values.summary.description
                )
                capsule.keywords = values.summary.keywords
                capsule.common_features = values.summary.common_features
                capsule.internal_variance = values.summary.internal_variance.value
                capsule.member_count = values.member_count
                capsule.average_membership_probability = values.average_membership_probability
                capsule.medoid_asset_id = medoid.asset_id
                # Kept as an API-compatible ID-only cache; the relation below is authoritative.
                capsule.representative_asset_ids = representative_ids

            await session.execute(
                delete(ClusterRepresentativeAsset).where(
                    ClusterRepresentativeAsset.cluster_capsule_id == capsule.cluster_capsule_id
                )
            )
            session.add_all(
                [
                    ClusterRepresentativeAsset(
                        cluster_capsule_id=capsule.cluster_capsule_id,
                        asset_id=item.asset_id,
                        role=item.role.value,
                        rank=item.rank,
                        distance_to_medoid=item.distance_to_medoid,
                        membership_probability=item.membership_probability,
                    )
                    for item in values.representatives
                ]
            )
            await session.flush()
            return _cluster_capsule_record(capsule)

    async def set_name_override(
        self,
        *,
        cluster_capsule_id: str,
        workspace_id: str,
        name: str | None,
    ) -> ClusterCapsuleRecord:
        """Set a user name, or clear it with ``None`` to restore the model value."""
        return await self.update_overrides(
            cluster_capsule_id=cluster_capsule_id,
            workspace_id=workspace_id,
            update_name=True,
            name=name,
            update_description=False,
            description=None,
        )

    async def set_description_override(
        self,
        *,
        cluster_capsule_id: str,
        workspace_id: str,
        description: str | None,
    ) -> ClusterCapsuleRecord:
        """Set a user description, or clear it with ``None`` to restore the model value."""
        return await self.update_overrides(
            cluster_capsule_id=cluster_capsule_id,
            workspace_id=workspace_id,
            update_name=False,
            name=None,
            update_description=True,
            description=description,
        )

    async def update_overrides(
        self,
        *,
        cluster_capsule_id: str,
        workspace_id: str,
        update_name: bool,
        name: str | None,
        update_description: bool,
        description: str | None,
    ) -> ClusterCapsuleRecord:
        """Apply one atomic front-end edit; ``None`` clears a supplied override."""
        if update_name:
            _validate_user_override(name, field="name")
        if update_description:
            _validate_user_override(description, field="description")
        async with self._database.session() as session, session.begin():
            capsule = await _load_capsule_for_update(
                session,
                cluster_capsule_id=cluster_capsule_id,
                workspace_id=workspace_id,
            )
            if update_name:
                capsule.user_override_name = name
                capsule.effective_name = name or capsule.model_generated_name
            if update_description:
                capsule.user_override_description = description
                capsule.effective_description = description or capsule.model_generated_description
            await session.flush()
            return _cluster_capsule_record(capsule)


def _validate_representatives(representatives: list[ClusterRepresentativeWrite]) -> None:
    ids = [item.asset_id for item in representatives]
    if len(ids) != len(set(ids)):
        raise ValueError("representative Asset IDs must be unique")
    if [item.rank for item in representatives] != list(range(len(representatives))):
        raise ValueError("representative ranks must be consecutive and start at zero")
    medoids = [item for item in representatives if item.role.value == "medoid"]
    if len(medoids) != 1 or medoids[0].rank != 0:
        raise ValueError("representatives must have exactly one rank-zero medoid")


@dataclass(slots=True, frozen=True)
class ClusterEmbeddingAsset:
    """The indexed vector and Asset metadata required for one clustering type."""

    embedding_id: str
    asset_id: str
    source_file_id: str
    asset_type: str
    asset_name: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    file_tree_context: list[str]
    source_relative_path: str = ""


@dataclass(slots=True, frozen=True)
class ClusterMembershipWrite:
    asset_id: str
    cluster_capsule_id: str | None
    hdbscan_label: int
    membership_probability: float
    is_noise: bool
    distance_to_representative: float | None


def _validate_user_override(value: str | None, *, field: str) -> None:
    if value is not None and not value.strip():
        raise ValueError(f"user override {field} must not be blank")


async def _load_representative_assets(
    session: AsyncSession,
    *,
    workspace_id: str,
    representative_ids: list[str],
) -> dict[str, Asset]:
    rows = await session.scalars(
        select(Asset)
        .where(
            Asset.workspace_id == workspace_id,
            Asset.asset_id.in_(representative_ids),
        )
        .with_for_update()
    )
    assets = {asset.asset_id: asset for asset in rows}
    missing = sorted(set(representative_ids) - set(assets))
    if missing:
        raise ValueError(f"representative Assets do not exist in workspace: {', '.join(missing)}")
    return assets


def _validate_representative_source_limits(
    representatives: list[ClusterRepresentativeWrite],
    assets: dict[str, Asset],
) -> None:
    counts: dict[str, int] = {}
    for representative in representatives:
        source_file_id = assets[representative.asset_id].source_file_id
        counts[source_file_id] = counts.get(source_file_id, 0) + 1
    if any(count > 2 for count in counts.values()):
        raise ValueError("at most two representative Assets may come from one source file")


async def _load_capsule_for_update(
    session: AsyncSession,
    *,
    cluster_capsule_id: str,
    workspace_id: str,
) -> ClusterCapsule:
    capsule = await session.scalar(
        select(ClusterCapsule)
        .where(
            ClusterCapsule.cluster_capsule_id == cluster_capsule_id,
            ClusterCapsule.workspace_id == workspace_id,
        )
        .with_for_update()
    )
    if capsule is None:
        raise ValueError(f"cluster capsule does not exist: {cluster_capsule_id}")
    return capsule


async def _load_run_for_update(session: AsyncSession, cluster_run_id: str) -> ClusterRun:
    run = await session.get(ClusterRun, cluster_run_id, with_for_update=True)
    if run is None:
        raise ValueError(f"cluster run does not exist: {cluster_run_id}")
    return run


def _cluster_capsule_record(capsule: ClusterCapsule) -> ClusterCapsuleRecord:
    return ClusterCapsuleRecord(
        cluster_capsule_id=capsule.cluster_capsule_id,
        cluster_run_id=capsule.cluster_run_id,
        workspace_id=capsule.workspace_id,
        embedding_type=capsule.embedding_type,
        cluster_label=capsule.cluster_label,
        model_generated_name=capsule.model_generated_name,
        user_override_name=capsule.user_override_name,
        effective_name=capsule.effective_name,
        model_generated_description=capsule.model_generated_description,
        user_override_description=capsule.user_override_description,
        effective_description=capsule.effective_description,
        keywords=list(capsule.keywords),
        common_features=list(capsule.common_features),
        internal_variance=(
            ClusterInternalVariance(capsule.internal_variance)
            if capsule.internal_variance is not None
            else None
        ),
        member_count=capsule.member_count,
        average_membership_probability=capsule.average_membership_probability,
        medoid_asset_id=capsule.medoid_asset_id,
        representative_asset_ids=list(capsule.representative_asset_ids),
        is_favorite=capsule.is_favorite,
    )


def _cluster_run_record(run: ClusterRun) -> ClusterRunRecord:
    return ClusterRunRecord(
        cluster_run_id=run.cluster_run_id,
        workspace_id=run.workspace_id,
        embedding_type=run.embedding_type,
        input_embedding_ids=list(run.input_embedding_ids),
        dataset_hash=run.dataset_hash,
        sample_count=run.sample_count,
        preprocessing=dict(run.preprocessing),
        parameters=dict(run.parameters),
        cluster_count=run.cluster_count,
        noise_count=run.noise_count,
        noise_ratio=run.noise_ratio,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )
