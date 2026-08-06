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
    ClusterExclusion,
    ClusterMembership,
    ClusterRepresentativeAsset,
    ClusterRun,
    CurrentCluster,
    CurrentClusterMember,
    EmbeddingRecord,
    ModelCallLog,
    ProcessingJob,
    SourceFile,
    Workspace,
)
from capsule.db.session import Database
from capsule.enums import (
    AssetIndexRole,
    AssetNameSource,
    AssetType,
    ClusterInternalVariance,
    ClusterMemberSource,
    ClusterMode,
    ClusterRunStatus,
    EmbeddingStatus,
    JobStatus,
    NewAssetClusterStatus,
    PipelineStage,
    ProcessingStatus,
)
from capsule.features import (
    embedding_channel_is_eligible,
    embedding_type_supports_asset_type,
)
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

    async def mark_import_upload_activity(
        self,
        *,
        job_id: str,
        workspace_id: str,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
                .with_for_update()
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status != JobStatus.QUEUED.value:
                raise ValueError(f"processing job cannot accept uploads from status {job.status}")
            job.updated_at = datetime.now(UTC)

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
        _validate_asset_hierarchy(assets, require_batch_parent=True)

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
            stored_by_key: dict[str, Asset] = {}

            def store(values: AssetCreate) -> Asset:
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
                stored_by_key[values.asset_key] = current
                return current

            # A child cannot be inserted until its persisted parent has a real ID:
            # the database check constraint deliberately disallows dangling children.
            for values in assets:
                if values.index_role != AssetIndexRole.CHILD:
                    store(values)

            await session.flush()
            for values in assets:
                if values.index_role != AssetIndexRole.CHILD:
                    continue
                current = store(values)
                _resolve_asset_parent(
                    asset=current,
                    values=values,
                    source_file_id=source_file_id,
                    possible_parents=stored_by_key,
                )

            await session.flush()

            for stale in existing.values():
                await session.delete(stale)

            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            await session.flush()
            return StoredFileResult(
                source_file_id=source_file_id,
                asset_ids=[stored_by_key[asset.asset_key].asset_id for asset in assets],
                indexable_asset_ids=[
                    stored_by_key[asset.asset_key].asset_id
                    for asset in assets
                    if asset.index_role != AssetIndexRole.PARENT
                ],
            )

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
        _validate_asset_hierarchy([asset], require_batch_parent=False)
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            possible_parents: dict[str, Asset] = {}
            if asset.index_role == AssetIndexRole.CHILD:
                parent = await session.scalar(
                    select(Asset)
                    .where(
                        Asset.source_file_id == source_file_id,
                        Asset.asset_key == asset.parent_asset_key,
                    )
                    .with_for_update()
                )
                if parent is not None:
                    possible_parents[asset.parent_asset_key or ""] = parent
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
            if asset.index_role == AssetIndexRole.CHILD:
                _resolve_asset_parent(
                    asset=current,
                    values=asset,
                    source_file_id=source_file_id,
                    possible_parents=possible_parents,
                )
            else:
                current.parent_asset_id = None
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
            stored_assets = list(
                await session.execute(
                    select(Asset.asset_id, Asset.index_role)
                    .where(
                        Asset.source_file_id == source_file_id,
                        Asset.generation == generation,
                    )
                    .order_by(Asset.asset_key)
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            asset_ids = [asset_id for asset_id, _ in stored_assets]
            return StoredFileResult(
                source_file_id=source_file_id,
                asset_ids=asset_ids,
                indexable_asset_ids=[
                    asset_id
                    for asset_id, index_role in stored_assets
                    if index_role != AssetIndexRole.PARENT.value
                ],
            )

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
                job.current_stage = PipelineStage.COMPLETED.value
            elif job.completed_count == 0:
                job.status = JobStatus.FAILED.value
                job.current_stage = PipelineStage.FAILED.value
            else:
                job.status = JobStatus.PARTIAL_FAILED.value
                job.current_stage = PipelineStage.COMPLETED.value
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

    async def clear_jobs(self, *, workspace_id: str) -> int:
        """Remove every processing-job record owned by one workspace."""
        async with self._database.session() as session, session.begin():
            deleted_ids = list(
                await session.scalars(
                    delete(ProcessingJob)
                    .where(ProcessingJob.workspace_id == workspace_id)
                    .returning(ProcessingJob.job_id)
                )
            )
            return len(deleted_ids)

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
        index_role=AssetIndexRole(asset.index_role),
        parent_asset_id=asset.parent_asset_id,
        child_order=asset.child_order,
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
    data["index_role"] = values.index_role.value
    data["processing_status"] = values.processing_status.value
    if values.index_role == AssetIndexRole.PARENT:
        data["processing_status"] = ProcessingStatus.COMPLETED.value
    data["source_contexts"] = [context.model_dump() for context in values.source_contexts]
    return data


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _update_asset(current: Asset, values: AssetCreate, *, content_changed: bool) -> None:
    current.workspace_id = values.workspace_id
    current.asset_type = values.asset_type.value
    current.file_name = values.file_name
    current.file_type = values.file_type
    current.index_role = values.index_role.value
    current.child_order = values.child_order
    if values.index_role != AssetIndexRole.CHILD:
        current.parent_asset_id = None
    current.content_hash = values.content_hash
    current.generation = values.generation
    current.file_tree_context = values.file_tree_context
    current.source_contexts = [context.model_dump() for context in values.source_contexts]
    current.file_info = values.file_info
    current.source_locator = values.source_locator
    current.raw_content = values.raw_content
    current.derived_file_uri = values.derived_file_uri
    current.preview_uri = values.preview_uri
    current.processing_status = (
        ProcessingStatus.COMPLETED.value
        if values.index_role == AssetIndexRole.PARENT
        else ProcessingStatus.PENDING.value
    )
    current.error_message = None
    if content_changed:
        if current.asset_name_source != AssetNameSource.USER.value:
            current.asset_name = None
            current.asset_name_source = None
        current.asset_description = None
        current.asset_features = {}
        current.feature_revision += 1
        current.embedding_revision += 1


def _validate_asset_hierarchy(
    assets: Sequence[AssetCreate],
    *,
    require_batch_parent: bool,
) -> None:
    """Reject invalid role/reference shapes before any row is changed."""
    by_key: dict[str, AssetCreate] = {}
    for asset in assets:
        if asset.asset_key in by_key:
            raise ValueError(f"duplicate asset_key in hierarchy batch: {asset.asset_key}")
        by_key[asset.asset_key] = asset
        if asset.index_role == AssetIndexRole.CHILD and not asset.parent_asset_key:
            raise ValueError("child assets require a parent_asset_key")

    if not require_batch_parent:
        return
    for asset in assets:
        if asset.index_role != AssetIndexRole.CHILD:
            continue
        parent = by_key.get(asset.parent_asset_key or "")
        if parent is None:
            raise ValueError("child asset parent does not exist in this batch")
        if parent.index_role != AssetIndexRole.PARENT:
            raise ValueError("child asset parent must have index_role=parent")


def _resolve_asset_parent(
    *,
    asset: Asset,
    values: AssetCreate,
    source_file_id: str,
    possible_parents: dict[str, Asset],
) -> None:
    """Link a child to a persisted parent in the same source file."""
    if values.index_role != AssetIndexRole.CHILD:
        asset.parent_asset_id = None
        return
    parent_key = values.parent_asset_key
    if parent_key is None:
        raise ValueError("child assets require a parent_asset_key")
    parent = possible_parents.get(parent_key)
    if parent is None:
        raise ValueError(f"child asset parent does not exist: {parent_key}")
    if parent.source_file_id != source_file_id:
        raise ValueError("child asset parent must belong to the same source file")
    if parent.index_role != AssetIndexRole.PARENT.value:
        raise ValueError("child asset parent must have index_role=parent")
    asset.parent_asset_id = parent.asset_id


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
    index_role: str = AssetIndexRole.STANDALONE.value


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
                Asset.index_role != AssetIndexRole.PARENT.value,
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
                index_role=asset.index_role,
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
        if asset.index_role == AssetIndexRole.PARENT.value:
            raise ValueError("parent assets must not receive embeddings")
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
                Asset.index_role != AssetIndexRole.PARENT.value,
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
#      Current cluster persistence
# ===========================================


@dataclass(slots=True, frozen=True)
class CurrentClusterRecord:
    cluster_id: str
    workspace_id: str
    embedding_type: str
    mode: ClusterMode
    name: str
    description: str
    representative_asset_id: str | None
    source_run_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class CurrentClusterMemberRecord:
    cluster_id: str
    asset_id: str
    embedding_type: str
    source: ClusterMemberSource
    score: float | None
    created_at: datetime


@dataclass(slots=True, frozen=True)
class CurrentClusterExclusionRecord:
    cluster_id: str
    asset_id: str
    created_by: str | None
    created_at: datetime


@dataclass(slots=True, frozen=True)
class CurrentClusterEmbedding:
    asset_id: str
    embedding_id: str


@dataclass(slots=True, frozen=True)
class ClusterBootstrapState:
    """Current channel size and full-run state used by bootstrap coordination."""

    has_baseline: bool
    run_in_progress: bool
    eligible_asset_count: int
    latest_run_id: str | None
    latest_sample_count: int | None


@dataclass(slots=True, frozen=True)
class NewAssetClusterStatusItemRecord:
    asset_id: str
    asset_type: AssetType
    file_name: str
    asset_name: str | None
    status: NewAssetClusterStatus
    cluster_id: str | None
    cluster_name: str | None
    cluster_mode: ClusterMode | None
    member_source: ClusterMemberSource | None
    score: float | None
    created_at: datetime


@dataclass(slots=True, frozen=True)
class NewAssetClusterStatusRecord:
    has_baseline: bool
    baseline_cluster_run_id: str | None
    baseline_sample_count: int | None
    eligible_asset_count: int
    items: tuple[NewAssetClusterStatusItemRecord, ...]


@dataclass(slots=True, frozen=True)
class _EligibleClusterAsset:
    embedding_id: str
    asset: Asset


@dataclass(slots=True, frozen=True)
class CurrentClusterMemberWrite:
    asset_id: str
    score: float | None = None


@dataclass(slots=True, frozen=True)
class CurrentClusterPublish:
    name: str
    description: str
    representative_asset_id: str | None
    members: Sequence[CurrentClusterMemberWrite]
    cluster_id: str | None = None


class CurrentClusterRepository:
    """Persist the currently effective clusters separately from immutable run history."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._new_cluster_id = id_factory("cluster")

    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: Sequence[ClusterMode] | None = None,
    ) -> list[CurrentClusterRecord]:
        statement = select(CurrentCluster).where(
            CurrentCluster.workspace_id == workspace_id,
            CurrentCluster.embedding_type == embedding_type,
        )
        if modes is not None:
            mode_values = [ClusterMode(mode).value for mode in modes]
            if not mode_values:
                return []
            statement = statement.where(CurrentCluster.mode.in_(mode_values))
        statement = statement.order_by(CurrentCluster.created_at, CurrentCluster.cluster_id)
        async with self._database.session() as session:
            rows = await session.scalars(statement)
            return [_current_cluster_record(cluster) for cluster in rows]

    async def get_cluster_bootstrap_state(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> ClusterBootstrapState:
        """Return enough durable state to decide whether initial clustering may start."""
        async with self._database.session() as session:
            latest_run = await _latest_cluster_baseline(
                session,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
            )
            run_in_progress = bool(
                await session.scalar(
                    select(func.count(ClusterRun.cluster_run_id)).where(
                        ClusterRun.workspace_id == workspace_id,
                        ClusterRun.embedding_type == embedding_type,
                        ClusterRun.status.in_(
                            [
                                ClusterRunStatus.PENDING.value,
                                ClusterRunStatus.RUNNING.value,
                            ]
                        ),
                    )
                )
            )
            eligible_assets = await _latest_eligible_cluster_assets(
                session,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                model_name=model_name,
                dimension=dimension,
                milvus_collection=milvus_collection,
            )
        return ClusterBootstrapState(
            has_baseline=latest_run is not None,
            run_in_progress=run_in_progress,
            eligible_asset_count=len(eligible_assets),
            latest_run_id=(latest_run.cluster_run_id if latest_run is not None else None),
            latest_sample_count=(latest_run.sample_count if latest_run is not None else None),
        )

    async def get_new_asset_cluster_status(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> NewAssetClusterStatusRecord:
        """Classify current eligible embeddings that are absent from the latest baseline."""
        async with self._database.session() as session:
            latest_run = await _latest_cluster_baseline(
                session,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
            )
            eligible_assets = await _latest_eligible_cluster_assets(
                session,
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                model_name=model_name,
                dimension=dimension,
                milvus_collection=milvus_collection,
            )
            baseline_embedding_ids = (
                set(latest_run.input_embedding_ids) if latest_run is not None else set()
            )
            new_assets = [
                item
                for item in eligible_assets
                if item.embedding_id not in baseline_embedding_ids
            ]
            new_asset_ids = [item.asset.asset_id for item in new_assets]
            member_rows = (
                (
                    await session.execute(
                        select(CurrentClusterMember, CurrentCluster)
                        .join(
                            CurrentCluster,
                            CurrentCluster.cluster_id == CurrentClusterMember.cluster_id,
                        )
                        .where(
                            CurrentCluster.workspace_id == workspace_id,
                            CurrentCluster.embedding_type == embedding_type,
                            CurrentClusterMember.asset_id.in_(new_asset_ids),
                        )
                    )
                ).all()
                if new_asset_ids
                else []
            )
        membership_by_asset = {
            member.asset_id: (member, cluster) for member, cluster in member_rows
        }
        items: list[NewAssetClusterStatusItemRecord] = []
        for eligible in sorted(
            new_assets,
            key=lambda item: (item.asset.created_at, item.asset.asset_id),
            reverse=True,
        ):
            asset = eligible.asset
            membership = membership_by_asset.get(asset.asset_id)
            member, cluster = membership if membership is not None else (None, None)
            if cluster is None:
                cluster_status = NewAssetClusterStatus.PENDING
            elif cluster.mode == ClusterMode.RESIDENT_MANUAL.value:
                cluster_status = NewAssetClusterStatus.MANUAL_MANAGEMENT
            else:
                cluster_status = NewAssetClusterStatus.INCREMENTALLY_CLUSTERED
            items.append(
                NewAssetClusterStatusItemRecord(
                    asset_id=asset.asset_id,
                    asset_type=AssetType(asset.asset_type),
                    file_name=asset.file_name,
                    asset_name=asset.asset_name,
                    status=cluster_status,
                    cluster_id=cluster.cluster_id if cluster is not None else None,
                    cluster_name=cluster.name if cluster is not None else None,
                    cluster_mode=(ClusterMode(cluster.mode) if cluster is not None else None),
                    member_source=(
                        ClusterMemberSource(member.source) if member is not None else None
                    ),
                    score=member.score if member is not None else None,
                    created_at=asset.created_at,
                )
            )
        return NewAssetClusterStatusRecord(
            has_baseline=latest_run is not None,
            baseline_cluster_run_id=(
                latest_run.cluster_run_id if latest_run is not None else None
            ),
            baseline_sample_count=(
                latest_run.sample_count if latest_run is not None else None
            ),
            eligible_asset_count=len(eligible_assets),
            items=tuple(items),
        )

    async def get_cluster(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> CurrentClusterRecord:
        async with self._database.session() as session:
            cluster = await session.scalar(
                select(CurrentCluster).where(
                    CurrentCluster.cluster_id == cluster_id,
                    CurrentCluster.workspace_id == workspace_id,
                )
            )
            if cluster is None:
                raise ValueError(f"current cluster does not exist: {cluster_id}")
            return _current_cluster_record(cluster)

    async def list_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> list[CurrentClusterMemberRecord]:
        async with self._database.session() as session:
            await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
            )
            rows = await session.scalars(
                select(CurrentClusterMember)
                .where(CurrentClusterMember.cluster_id == cluster_id)
                .order_by(CurrentClusterMember.created_at, CurrentClusterMember.asset_id)
            )
            return [_current_cluster_member_record(member) for member in rows]

    async def list_resident_asset_ids(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
    ) -> set[str]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(CurrentClusterMember.asset_id)
                .join(CurrentCluster, CurrentCluster.cluster_id == CurrentClusterMember.cluster_id)
                .where(
                    CurrentCluster.workspace_id == workspace_id,
                    CurrentCluster.embedding_type == embedding_type,
                    CurrentCluster.mode.in_(
                        [ClusterMode.RESIDENT_OPEN.value, ClusterMode.RESIDENT_MANUAL.value]
                    ),
                )
            )
            return set(rows)

    async def set_mode(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        mode: ClusterMode,
    ) -> CurrentClusterRecord:
        parsed_mode = ClusterMode(mode)
        async with self._database.session() as session, session.begin():
            cluster = await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                for_update=True,
            )
            cluster.mode = parsed_mode.value
            await session.flush()
            await session.refresh(cluster)
            return _current_cluster_record(cluster)

    async def set_name(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        name: str,
    ) -> CurrentClusterRecord:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("current cluster name cannot be empty")
        async with self._database.session() as session, session.begin():
            cluster = await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                for_update=True,
            )
            cluster.name = normalized_name
            await session.flush()
            await session.refresh(cluster)
            return _current_cluster_record(cluster)

    async def attach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: Sequence[str],
        source: ClusterMemberSource = ClusterMemberSource.USER,
        scores: dict[str, float] | None = None,
    ) -> list[CurrentClusterMemberRecord]:
        """Atomically attach Assets; algorithmic writes never override existing assignments."""
        requested_ids = list(dict.fromkeys(asset_ids))
        if not requested_ids:
            return []
        parsed_source = ClusterMemberSource(source)
        unknown_scores = set(scores or {}) - set(requested_ids)
        if unknown_scores:
            raise ValueError("scores contain Assets that were not requested")

        async with self._database.session() as session, session.begin():
            cluster = await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                for_update=True,
            )
            if (
                parsed_source is ClusterMemberSource.USER
                and cluster.mode == ClusterMode.DYNAMIC.value
            ):
                raise ValueError("manual assignments require a resident cluster")
            if (
                parsed_source is not ClusterMemberSource.USER
                and cluster.mode == ClusterMode.RESIDENT_MANUAL.value
            ):
                raise ValueError("resident_manual clusters only accept user assignments")
            await _validate_indexed_assets(
                session,
                workspace_id=workspace_id,
                embedding_type=cluster.embedding_type,
                asset_ids=requested_ids,
            )

            current_rows = list(
                await session.scalars(
                    select(CurrentClusterMember)
                    .where(
                        CurrentClusterMember.asset_id.in_(requested_ids),
                        CurrentClusterMember.embedding_type == cluster.embedding_type,
                    )
                    .with_for_update()
                )
            )
            current_by_asset = {member.asset_id: member for member in current_rows}
            exclusion_rows = list(
                await session.scalars(
                    select(ClusterExclusion)
                    .where(
                        ClusterExclusion.cluster_id == cluster_id,
                        ClusterExclusion.asset_id.in_(requested_ids),
                    )
                    .with_for_update()
                )
            )
            exclusions = {item.asset_id: item for item in exclusion_rows}

            attached: dict[str, CurrentClusterMember] = {}
            for asset_id in requested_ids:
                existing = current_by_asset.get(asset_id)
                exclusion = exclusions.get(asset_id)
                if parsed_source is not ClusterMemberSource.USER:
                    if exclusion is not None or existing is not None:
                        if existing is not None and existing.cluster_id == cluster_id:
                            attached[asset_id] = existing
                        continue
                else:
                    if exclusion is not None:
                        await session.delete(exclusion)
                    if existing is not None:
                        if existing.cluster_id == cluster_id:
                            existing.source = parsed_source.value
                            existing.score = (scores or {}).get(asset_id)
                            attached[asset_id] = existing
                            continue
                        await session.delete(existing)
                        await session.flush()

                member = CurrentClusterMember(
                    cluster_id=cluster_id,
                    asset_id=asset_id,
                    embedding_type=cluster.embedding_type,
                    source=parsed_source.value,
                    score=(scores or {}).get(asset_id),
                )
                session.add(member)
                attached[asset_id] = member

            await session.flush()
            return [
                _current_cluster_member_record(attached[asset_id])
                for asset_id in requested_ids
                if asset_id in attached
            ]

    async def detach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: Sequence[str],
        created_by: str | None = None,
    ) -> list[str]:
        """Remove current members and add a durable do-not-reenter rule."""
        requested_ids = list(dict.fromkeys(asset_ids))
        if not requested_ids:
            return []
        async with self._database.session() as session, session.begin():
            await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                for_update=True,
            )
            members = list(
                await session.scalars(
                    select(CurrentClusterMember)
                    .where(
                        CurrentClusterMember.cluster_id == cluster_id,
                        CurrentClusterMember.asset_id.in_(requested_ids),
                    )
                    .with_for_update()
                )
            )
            member_by_asset = {member.asset_id: member for member in members}
            detached_ids = [asset_id for asset_id in requested_ids if asset_id in member_by_asset]
            if not detached_ids:
                return []

            existing_exclusions = set(
                await session.scalars(
                    select(ClusterExclusion.asset_id).where(
                        ClusterExclusion.cluster_id == cluster_id,
                        ClusterExclusion.asset_id.in_(detached_ids),
                    )
                )
            )
            for asset_id in detached_ids:
                await session.delete(member_by_asset[asset_id])
                if asset_id not in existing_exclusions:
                    session.add(
                        ClusterExclusion(
                            cluster_id=cluster_id,
                            asset_id=asset_id,
                            created_by=created_by,
                        )
                    )
            return detached_ids

    async def list_exclusions(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: Sequence[str] | None = None,
    ) -> list[CurrentClusterExclusionRecord]:
        async with self._database.session() as session:
            await _load_current_cluster(
                session,
                cluster_id=cluster_id,
                workspace_id=workspace_id,
            )
            statement = select(ClusterExclusion).where(
                ClusterExclusion.cluster_id == cluster_id
            )
            if asset_ids is not None:
                requested_ids = list(dict.fromkeys(asset_ids))
                if not requested_ids:
                    return []
                statement = statement.where(ClusterExclusion.asset_id.in_(requested_ids))
            rows = await session.scalars(
                statement.order_by(ClusterExclusion.created_at, ClusterExclusion.asset_id)
            )
            return [_current_cluster_exclusion_record(exclusion) for exclusion in rows]

    async def list_excluded_pairs(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        cluster_ids: Sequence[str],
        asset_ids: Sequence[str],
    ) -> set[tuple[str, str]]:
        requested_clusters = list(dict.fromkeys(cluster_ids))
        requested_assets = list(dict.fromkeys(asset_ids))
        if not requested_clusters or not requested_assets:
            return set()
        async with self._database.session() as session:
            rows = (
                await session.execute(
                    select(ClusterExclusion.cluster_id, ClusterExclusion.asset_id)
                    .join(CurrentCluster, CurrentCluster.cluster_id == ClusterExclusion.cluster_id)
                    .where(
                        CurrentCluster.workspace_id == workspace_id,
                        CurrentCluster.embedding_type == embedding_type,
                        ClusterExclusion.cluster_id.in_(requested_clusters),
                        ClusterExclusion.asset_id.in_(requested_assets),
                    )
                )
            ).all()
            return {(cluster_id, asset_id) for cluster_id, asset_id in rows}

    async def list_indexed_asset_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        asset_ids: Sequence[str],
    ) -> list[CurrentClusterEmbedding]:
        requested_ids = list(dict.fromkeys(asset_ids))
        if not requested_ids:
            return []
        statement = (
            select(EmbeddingRecord.asset_id, EmbeddingRecord.embedding_id)
            .join(Asset, Asset.asset_id == EmbeddingRecord.asset_id)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                EmbeddingRecord.workspace_id == workspace_id,
                EmbeddingRecord.embedding_type == embedding_type,
                EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
                EmbeddingRecord.asset_id.in_(requested_ids),
                Asset.generation == SourceFile.processing_generation,
                Asset.index_role != AssetIndexRole.PARENT.value,
            )
            .order_by(EmbeddingRecord.created_at.desc(), EmbeddingRecord.embedding_id.desc())
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        latest: dict[str, CurrentClusterEmbedding] = {}
        for asset_id, embedding_id in rows:
            latest.setdefault(
                asset_id,
                CurrentClusterEmbedding(asset_id=asset_id, embedding_id=embedding_id),
            )
        return [latest[asset_id] for asset_id in requested_ids if asset_id in latest]

    async def publish_dynamic_clusters(
        self,
        *,
        run_id: str,
        workspace_id: str,
        embedding_type: str,
        clusters: Sequence[CurrentClusterPublish],
    ) -> list[CurrentClusterRecord]:
        """Atomically replace only one dimension's dynamic current clusters."""
        published = list(clusters)
        _validate_current_cluster_publish(published)
        async with self._database.session() as session, session.begin():
            run = await session.get(ClusterRun, run_id, with_for_update=True)
            if run is None:
                raise ValueError(f"cluster run does not exist: {run_id}")
            if run.workspace_id != workspace_id or run.embedding_type != embedding_type:
                raise ValueError("cluster run workspace or embedding type does not match publish")

            all_asset_ids = [member.asset_id for item in published for member in item.members]
            representative_ids = [
                item.representative_asset_id
                for item in published
                if item.representative_asset_id is not None
            ]
            await _validate_workspace_assets(
                session,
                workspace_id=workspace_id,
                asset_ids=[*all_asset_ids, *representative_ids],
            )

            resident_asset_ids = set(
                await session.scalars(
                    select(CurrentClusterMember.asset_id)
                    .join(
                        CurrentCluster,
                        CurrentCluster.cluster_id == CurrentClusterMember.cluster_id,
                    )
                    .where(
                        CurrentCluster.workspace_id == workspace_id,
                        CurrentCluster.embedding_type == embedding_type,
                        CurrentCluster.mode.in_(
                            [ClusterMode.RESIDENT_OPEN.value, ClusterMode.RESIDENT_MANUAL.value]
                        ),
                    )
                    .with_for_update()
                )
            )
            overlap = sorted(resident_asset_ids.intersection(all_asset_ids))
            if overlap:
                raise ValueError(
                    "dynamic publish contains resident Assets: " + ", ".join(overlap)
                )

            await session.execute(
                delete(CurrentCluster).where(
                    CurrentCluster.workspace_id == workspace_id,
                    CurrentCluster.embedding_type == embedding_type,
                    CurrentCluster.mode == ClusterMode.DYNAMIC.value,
                )
            )

            stored: list[CurrentCluster] = []
            for item in published:
                cluster = CurrentCluster(
                    cluster_id=item.cluster_id or self._new_cluster_id(),
                    workspace_id=workspace_id,
                    embedding_type=embedding_type,
                    mode=ClusterMode.DYNAMIC.value,
                    name=item.name,
                    description=item.description,
                    representative_asset_id=item.representative_asset_id,
                    source_run_id=run_id,
                )
                session.add(cluster)
                stored.append(cluster)
                session.add_all(
                    [
                        CurrentClusterMember(
                            cluster_id=cluster.cluster_id,
                            asset_id=member.asset_id,
                            embedding_type=embedding_type,
                            source=ClusterMemberSource.FULL_CLUSTER.value,
                            score=member.score,
                        )
                        for member in item.members
                    ]
                )
            await session.flush()
            return [_current_cluster_record(cluster) for cluster in stored]


# ===========================================
#      Cluster Capsule persistence
# ===========================================


class ClusterRepository:
    """Persist model summaries, representative Asset IDs, and user overrides."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def create_pending_run(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        preprocessing: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> str:
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
                preprocessing={
                    "submission_mode": "async_api",
                    **(preprocessing or {}),
                },
                parameters=dict(parameters or {}),
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


async def _latest_cluster_baseline(
    session: AsyncSession,
    *,
    workspace_id: str,
    embedding_type: str,
) -> ClusterRun | None:
    return cast(
        ClusterRun | None,
        await session.scalar(
            select(ClusterRun)
            .where(
                ClusterRun.workspace_id == workspace_id,
                ClusterRun.embedding_type == embedding_type,
                ClusterRun.status.in_(
                    [
                        ClusterRunStatus.COMPLETED.value,
                        ClusterRunStatus.INSUFFICIENT_DATA.value,
                    ]
                ),
            )
            .order_by(
                ClusterRun.completed_at.desc().nullslast(),
                ClusterRun.started_at.desc().nullslast(),
                ClusterRun.cluster_run_id.desc(),
            )
            .limit(1)
        ),
    )


async def _latest_eligible_cluster_assets(
    session: AsyncSession,
    *,
    workspace_id: str,
    embedding_type: str,
    model_name: str,
    dimension: int,
    milvus_collection: str,
) -> list[_EligibleClusterAsset]:
    rows = (
        await session.execute(
            select(EmbeddingRecord, Asset)
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
                Asset.index_role != AssetIndexRole.PARENT.value,
            )
            .order_by(
                EmbeddingRecord.created_at.desc(),
                EmbeddingRecord.embedding_id.desc(),
            )
        )
    ).all()
    selected: dict[str, _EligibleClusterAsset] = {}
    for embedding, asset in rows:
        if asset.asset_id in selected:
            continue
        if not embedding_type_supports_asset_type(
            embedding_type=embedding_type,
            asset_type=asset.asset_type,
        ):
            continue
        if not embedding_channel_is_eligible(
            embedding_type=embedding_type,
            asset_features=asset.asset_features,
            asset_description=asset.asset_description,
        ):
            continue
        selected[asset.asset_id] = _EligibleClusterAsset(
            embedding_id=embedding.embedding_id,
            asset=asset,
        )
    return list(selected.values())


def _current_cluster_record(cluster: CurrentCluster) -> CurrentClusterRecord:
    return CurrentClusterRecord(
        cluster_id=cluster.cluster_id,
        workspace_id=cluster.workspace_id,
        embedding_type=cluster.embedding_type,
        mode=ClusterMode(cluster.mode),
        name=cluster.name,
        description=cluster.description,
        representative_asset_id=cluster.representative_asset_id,
        source_run_id=cluster.source_run_id,
        created_at=cluster.created_at,
        updated_at=cluster.updated_at,
    )


def _current_cluster_member_record(
    member: CurrentClusterMember,
) -> CurrentClusterMemberRecord:
    return CurrentClusterMemberRecord(
        cluster_id=member.cluster_id,
        asset_id=member.asset_id,
        embedding_type=member.embedding_type,
        source=ClusterMemberSource(member.source),
        score=member.score,
        created_at=member.created_at,
    )


def _current_cluster_exclusion_record(
    exclusion: ClusterExclusion,
) -> CurrentClusterExclusionRecord:
    return CurrentClusterExclusionRecord(
        cluster_id=exclusion.cluster_id,
        asset_id=exclusion.asset_id,
        created_by=exclusion.created_by,
        created_at=exclusion.created_at,
    )


async def _load_current_cluster(
    session: AsyncSession,
    *,
    cluster_id: str,
    workspace_id: str,
    for_update: bool = False,
) -> CurrentCluster:
    statement = select(CurrentCluster).where(
        CurrentCluster.cluster_id == cluster_id,
        CurrentCluster.workspace_id == workspace_id,
    )
    if for_update:
        statement = statement.with_for_update()
    cluster = await session.scalar(statement)
    if cluster is None:
        raise ValueError(f"current cluster does not exist: {cluster_id}")
    return cluster


async def _validate_workspace_assets(
    session: AsyncSession,
    *,
    workspace_id: str,
    asset_ids: Sequence[str],
) -> None:
    requested_ids = set(asset_ids)
    if not requested_ids:
        return
    rows = await session.scalars(
        select(Asset.asset_id)
        .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
        .where(
            Asset.workspace_id == workspace_id,
            Asset.asset_id.in_(requested_ids),
            Asset.generation == SourceFile.processing_generation,
        )
        .with_for_update()
    )
    missing = sorted(requested_ids - set(rows))
    if missing:
        raise ValueError("Assets do not exist in workspace: " + ", ".join(missing))


async def _validate_indexed_assets(
    session: AsyncSession,
    *,
    workspace_id: str,
    embedding_type: str,
    asset_ids: Sequence[str],
) -> None:
    await _validate_workspace_assets(
        session,
        workspace_id=workspace_id,
        asset_ids=asset_ids,
    )
    indexed_ids = set(
        await session.scalars(
            select(EmbeddingRecord.asset_id)
            .where(
                EmbeddingRecord.workspace_id == workspace_id,
                EmbeddingRecord.embedding_type == embedding_type,
                EmbeddingRecord.asset_id.in_(asset_ids),
                EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
            )
            .distinct()
        )
    )
    missing = sorted(set(asset_ids) - indexed_ids)
    if missing:
        raise ValueError(
            f"Assets do not have indexed {embedding_type} embeddings: " + ", ".join(missing)
        )


def _validate_current_cluster_publish(clusters: Sequence[CurrentClusterPublish]) -> None:
    cluster_ids = [item.cluster_id for item in clusters if item.cluster_id is not None]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("published cluster IDs must be unique")
    all_asset_ids: list[str] = []
    for item in clusters:
        if not item.name.strip():
            raise ValueError("published cluster name must not be blank")
        if not item.description.strip():
            raise ValueError("published cluster description must not be blank")
        member_ids = [member.asset_id for member in item.members]
        if not member_ids:
            raise ValueError("published dynamic clusters must contain at least one Asset")
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("published cluster member IDs must be unique")
        if (
            item.representative_asset_id is not None
            and item.representative_asset_id not in member_ids
        ):
            raise ValueError("representative Asset must be a member of its published cluster")
        all_asset_ids.extend(member_ids)
    if len(all_asset_ids) != len(set(all_asset_ids)):
        raise ValueError("an Asset may only appear in one published cluster per embedding type")


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
