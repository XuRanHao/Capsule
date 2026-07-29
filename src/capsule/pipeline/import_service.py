"""Browser folder import staging, with one retriable request per source file."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from fastapi import UploadFile
from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository
from capsule.enums import EmbeddingType, JobStatus, PipelineStage
from capsule.parsers.discovery import SUPPORTED_EXTENSIONS, discover_files
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.runner import PipelineRunner, PipelineRunResult
from capsule.pipeline.understanding import AssetUnderstandingService


class ImportSubmissionError(ValueError):
    """The browser import request cannot be accepted in its current state."""


class ImportFileTooLargeError(ImportSubmissionError):
    """A single browser-selected file exceeds the configured import limit."""


@dataclass(slots=True, frozen=True)
class BrowserImportJob:
    job_id: str
    staged_path: Path


@dataclass(slots=True, frozen=True)
class ImportCompletion:
    job_id: str
    staged_path: Path
    file_count: int


class AssetEnrichmentResult(BaseModel):
    job_id: str
    workspace_id: str
    requested_asset_count: int
    completed_asset_count: int
    partial_failed_asset_count: int
    errors: list[dict[str, str]] = Field(default_factory=list)


async def enrich_assets(
    *,
    job_id: str,
    workspace_id: str,
    asset_ids: list[str],
    repository: AssetRepository,
    understanding_service: AssetUnderstandingService,
    embedding_service: AssetEmbeddingService,
) -> AssetEnrichmentResult:
    """Run model understanding and every independent embedding channel."""
    await repository.begin_asset_enrichment(asset_ids=asset_ids)
    errors: list[dict[str, str]] = []
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.UNDERSTANDING,
    )
    understanding = await understanding_service.run(
        workspace_id=workspace_id,
        asset_ids=asset_ids,
    )
    stage_durations_ms = {
        PipelineStage.UNDERSTANDING.value: understanding.understanding_duration_ms,
        PipelineStage.FEATURE_READY.value: understanding.feature_ready_duration_ms,
        PipelineStage.EMBEDDING.value: 0.0,
        PipelineStage.INDEXING.value: 0.0,
    }
    errors.extend(
        {
            "asset_id": error["asset_id"],
            "stage": PipelineStage.UNDERSTANDING.value,
            "error": error["error"],
        }
        for error in understanding.errors
    )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.FEATURE_READY,
    )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.EMBEDDING,
    )
    for embedding_type in EmbeddingType:
        embedding = await embedding_service.run(
            workspace_id=workspace_id,
            embedding_type=embedding_type,
            asset_ids=asset_ids,
        )
        stage_durations_ms[
            PipelineStage.EMBEDDING.value
        ] += embedding.embedding_duration_ms
        stage_durations_ms[
            PipelineStage.INDEXING.value
        ] += embedding.indexing_duration_ms
        errors.extend(
            {
                "asset_id": error["asset_id"],
                "stage": f"{PipelineStage.EMBEDDING.value}:{embedding_type.value}",
                "error": error["error"],
            }
            for error in embedding.errors
        )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.INDEXING,
    )
    await repository.add_job_stage_durations(
        job_id=job_id,
        durations_ms=stage_durations_ms,
    )
    await repository.finalize_enrichment(
        job_id=job_id,
        asset_ids=asset_ids,
        errors=errors,
    )
    failed_asset_ids = {error["asset_id"] for error in errors}
    return AssetEnrichmentResult(
        job_id=job_id,
        workspace_id=workspace_id,
        requested_asset_count=len(asset_ids),
        completed_asset_count=len(asset_ids) - len(failed_asset_ids),
        partial_failed_asset_count=len(failed_asset_ids),
        errors=errors,
    )


class BrowserImportService:
    """Own the browser upload lifecycle before delegating to ``PipelineRunner``.

    A job is created first, then every file is uploaded independently with its
    browser-relative path.  This keeps retries scoped to one source file and
    prevents the assetization pipeline from seeing a half-uploaded folder.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        repository: AssetRepository,
        runner: PipelineRunner,
        understanding_service: AssetUnderstandingService | None = None,
        embedding_service: AssetEmbeddingService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._runner = runner
        self._understanding_service = understanding_service
        self._embedding_service = embedding_service

    async def create_job(self, *, workspace_id: str) -> BrowserImportJob:
        root = self._settings.import_root.expanduser().resolve()
        job_id = await self._repository.create_pending_import_job(
            workspace_id=workspace_id,
            import_root=root,
        )
        staged_path = root / job_id
        try:
            await asyncio.to_thread(staged_path.mkdir, parents=True, exist_ok=False)
        except Exception as exc:
            await self._repository.fail_job(job_id=job_id, error=str(exc) or type(exc).__name__)
            raise
        return BrowserImportJob(job_id=job_id, staged_path=staged_path)

    async def upload_file(
        self,
        *,
        job_id: str,
        workspace_id: str,
        file: UploadFile,
        relative_path: str,
    ) -> int:
        job = await self._repository.get_job(job_id=job_id, workspace_id=workspace_id)
        if job.status != JobStatus.QUEUED.value:
            raise ImportSubmissionError("files can only be uploaded while the import job is queued")

        staged_path = self._job_staging_path(job.input_path)
        target = staged_path / _validated_relative_path(file, relative_path)
        if not staged_path.is_dir():
            raise ImportSubmissionError("import staging directory is unavailable")
        try:
            return await asyncio.to_thread(
                _copy_upload_atomically,
                file.file,
                target,
                self._settings.import_file_max_bytes,
            )
        except ImportFileTooLargeError:
            raise
        except OSError as exc:
            raise ImportSubmissionError(str(exc) or "failed to stage uploaded file") from exc

    async def complete_job(
        self,
        *,
        job_id: str,
        workspace_id: str,
    ) -> ImportCompletion:
        job = await self._repository.get_job(job_id=job_id, workspace_id=workspace_id)
        if job.status != JobStatus.QUEUED.value:
            raise ImportSubmissionError("import job has already been started")
        staged_path = self._job_staging_path(job.input_path)
        file_count = len(discover_files(staged_path))
        if file_count == 0:
            raise ImportSubmissionError(
                "at least one supported file must be uploaded before completion"
            )
        await self._repository.start_import_job(job_id=job_id, total_count=file_count)
        return ImportCompletion(
            job_id=job_id,
            staged_path=staged_path,
            file_count=file_count,
        )

    async def execute(
        self,
        *,
        completion: ImportCompletion,
        workspace_id: str,
    ) -> PipelineRunResult | None:
        try:
            result = await self._runner.run(
                completion.staged_path,
                workspace_id,
                job_id=completion.job_id,
            )
            asset_ids = list(getattr(result, "asset_ids", []))
            if (
                not asset_ids
                or self._understanding_service is None
                or self._embedding_service is None
            ):
                return result

            await enrich_assets(
                job_id=completion.job_id,
                workspace_id=workspace_id,
                asset_ids=asset_ids,
                repository=self._repository,
                understanding_service=self._understanding_service,
                embedding_service=self._embedding_service,
            )
            return result
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            await self._repository.fail_job(job_id=completion.job_id, error=message)
            return None

    def _job_staging_path(self, input_path: str) -> Path:
        root = self._settings.import_root.expanduser().resolve()
        staging_path = Path(input_path).resolve()
        try:
            staging_path.relative_to(root)
        except ValueError as exc:
            raise ImportSubmissionError("import job has an invalid staging path") from exc
        return staging_path


def _validated_relative_path(file: UploadFile, raw_path: str) -> Path:
    path = _safe_relative_path(raw_path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImportSubmissionError(f"unsupported file extension: {path.suffix.lower()}")
    if file.filename and path.name != Path(file.filename).name:
        raise ImportSubmissionError("relative path file name does not match uploaded file")
    return path


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.parts or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ImportSubmissionError("relative_path must be a safe non-empty relative path")
    return Path(*path.parts)


def _copy_upload_atomically(source: BinaryIO, target: Path, max_bytes: int) -> int:
    """Write one upload through a sibling temporary file, then replace atomically."""
    source.seek(0)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.uploading")
    total = 0
    try:
        with temporary.open("xb") as destination:
            while chunk := source.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ImportFileTooLargeError(f"file exceeds {max_bytes} bytes")
                destination.write(chunk)
        os.replace(temporary, target)
        return total
    finally:
        if temporary.exists():
            temporary.unlink()
