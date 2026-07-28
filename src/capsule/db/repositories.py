"""Transactional persistence for source files, assets, and processing jobs."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.db.models import Asset, ProcessingJob, SourceFile, Workspace
from capsule.db.session import Database
from capsule.enums import AssetNameSource, JobStatus, PipelineStage, ProcessingStatus
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
