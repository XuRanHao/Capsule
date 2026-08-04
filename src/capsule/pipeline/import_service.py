"""Browser folder import staging, with one retriable request per source file."""

import asyncio
import os
import time
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


@dataclass(slots=True)
class _EnrichmentBatchResult:
    errors: list[dict[str, str]]
    stage_durations_ms: dict[str, float]


async def enrich_assets(
    *,
    job_id: str,
    workspace_id: str,
    asset_ids: list[str],
    repository: AssetRepository,
    understanding_service: AssetUnderstandingService,
    embedding_service: AssetEmbeddingService,
    force_understanding: bool = False,
) -> AssetEnrichmentResult:
    """Overlap native embedding with understanding, then fan out text channels."""
    await repository.begin_asset_enrichment(asset_ids=asset_ids)
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.UNDERSTANDING,
    )
    batch = await _run_enrichment_batch(
        workspace_id=workspace_id,
        asset_ids=asset_ids,
        understanding_service=understanding_service,
        embedding_service=embedding_service,
        force_understanding=force_understanding,
    )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.FEATURE_READY,
    )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.EMBEDDING,
    )
    await repository.set_job_stage(
        job_id=job_id,
        stage=PipelineStage.INDEXING,
    )
    await repository.add_job_stage_durations(
        job_id=job_id,
        durations_ms=batch.stage_durations_ms,
    )
    await repository.finalize_enrichment(
        job_id=job_id,
        asset_ids=asset_ids,
        errors=batch.errors,
    )
    failed_asset_ids = {error["asset_id"] for error in batch.errors}
    return AssetEnrichmentResult(
        job_id=job_id,
        workspace_id=workspace_id,
        requested_asset_count=len(asset_ids),
        completed_asset_count=len(asset_ids) - len(failed_asset_ids),
        partial_failed_asset_count=len(failed_asset_ids),
        errors=batch.errors,
    )


async def _run_enrichment_batch(
    *,
    workspace_id: str,
    asset_ids: list[str],
    understanding_service: AssetUnderstandingService,
    embedding_service: AssetEmbeddingService,
    force_understanding: bool = False,
) -> _EnrichmentBatchResult:
    """Enrich a committed Asset batch without mutating aggregate Job state."""
    errors: list[dict[str, str]] = []
    native_embedding_task = asyncio.create_task(
        embedding_service.run(
            workspace_id=workspace_id,
            embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
            asset_ids=asset_ids,
        )
    )
    try:
        understanding = await understanding_service.run(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
            force=force_understanding,
        )
    except BaseException:
        native_embedding_task.cancel()
        await asyncio.gather(native_embedding_task, return_exceptions=True)
        raise
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
    text_embedding_types = [
        embedding_type
        for embedding_type in EmbeddingType
        if embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
    ]
    embedding_wait_started = time.perf_counter()
    native_embedding, text_embeddings = await asyncio.gather(
        native_embedding_task,
        embedding_service.run_many(
            workspace_id=workspace_id,
            embedding_types=text_embedding_types,
            asset_ids=asset_ids,
        ),
    )
    embedding_elapsed_ms = (time.perf_counter() - embedding_wait_started) * 1000
    embeddings = [native_embedding, *text_embeddings]
    model_weight = sum(embedding.embedding_duration_ms for embedding in embeddings)
    indexing_weight = sum(embedding.indexing_duration_ms for embedding in embeddings)
    measured_weight = model_weight + indexing_weight
    if measured_weight:
        stage_durations_ms[PipelineStage.EMBEDDING.value] = (
            embedding_elapsed_ms * model_weight / measured_weight
        )
        stage_durations_ms[PipelineStage.INDEXING.value] = max(
            0.0,
            embedding_elapsed_ms - stage_durations_ms[PipelineStage.EMBEDDING.value],
        )
    else:
        stage_durations_ms[PipelineStage.EMBEDDING.value] = embedding_elapsed_ms

    for embedding in embeddings:
        errors.extend(
            {
                "asset_id": error["asset_id"],
                "stage": (f"{PipelineStage.EMBEDDING.value}:{embedding.embedding_type}"),
                "error": error["error"],
            }
            for error in embedding.errors
        )
    return _EnrichmentBatchResult(
        errors=errors,
        stage_durations_ms=stage_durations_ms,
    )


class AssetEnrichmentPipeline:
    """Consume committed Assets immediately through a bounded worker queue."""

    def __init__(
        self,
        *,
        settings: Settings,
        job_id: str,
        workspace_id: str,
        repository: AssetRepository,
        understanding_service: AssetUnderstandingService,
        embedding_service: AssetEmbeddingService,
    ) -> None:
        self._job_id = job_id
        self._workspace_id = workspace_id
        self._repository = repository
        self._understanding_service = understanding_service
        self._embedding_service = embedding_service
        self._worker_count = settings.understanding_concurrency
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(
            maxsize=settings.asset_enrichment_queue_size
        )
        self._workers: list[asyncio.Task[None]] = []
        self._submission_lock = asyncio.Lock()
        self._seen_asset_ids: set[str] = set()
        self._asset_ids: list[str] = []
        self._outcomes: list[_EnrichmentBatchResult] = []
        self._stage_started = False
        self._started = False
        self._closed = False
        self._active_count = 0
        self._active_started_at = 0.0
        self._active_elapsed_ms = 0.0

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("asset enrichment pipeline has already started")
        self._started = True
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self._worker_count)]

    async def submit(self, asset_ids: list[str]) -> None:
        """Queue newly committed Assets, applying backpressure at the configured bound."""
        if not self._started or self._closed:
            raise RuntimeError("asset enrichment pipeline is not accepting Assets")
        async with self._submission_lock:
            pending = [
                asset_id
                for asset_id in dict.fromkeys(asset_ids)
                if asset_id not in self._seen_asset_ids
            ]
            if not pending:
                return
            if not self._stage_started:
                await self._repository.set_job_stage(
                    job_id=self._job_id,
                    stage=PipelineStage.UNDERSTANDING,
                )
                self._stage_started = True
            for asset_id in pending:
                self._seen_asset_ids.add(asset_id)
                self._asset_ids.append(asset_id)
                await self._queue.put(asset_id)

    async def finish(self) -> AssetEnrichmentResult:
        if not self._started or self._closed:
            raise RuntimeError("asset enrichment pipeline cannot be finished")
        await self._queue.join()
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers)
        self._closed = True

        if not self._asset_ids:
            await self._repository.finalize_job(job_id=self._job_id)
            return AssetEnrichmentResult(
                job_id=self._job_id,
                workspace_id=self._workspace_id,
                requested_asset_count=0,
                completed_asset_count=0,
                partial_failed_asset_count=0,
            )

        errors = [error for outcome in self._outcomes for error in outcome.errors]
        stage_durations_ms = self._normalized_stage_durations()
        for stage in (
            PipelineStage.FEATURE_READY,
            PipelineStage.EMBEDDING,
            PipelineStage.INDEXING,
        ):
            await self._repository.set_job_stage(job_id=self._job_id, stage=stage)
        await self._repository.add_job_stage_durations(
            job_id=self._job_id,
            durations_ms=stage_durations_ms,
        )
        await self._repository.finalize_enrichment(
            job_id=self._job_id,
            asset_ids=self._asset_ids,
            errors=errors,
        )
        failed_asset_ids = {error["asset_id"] for error in errors}
        return AssetEnrichmentResult(
            job_id=self._job_id,
            workspace_id=self._workspace_id,
            requested_asset_count=len(self._asset_ids),
            completed_asset_count=len(self._asset_ids) - len(failed_asset_ids),
            partial_failed_asset_count=len(failed_asset_ids),
            errors=errors,
        )

    async def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker(self) -> None:
        while True:
            asset_id = await self._queue.get()
            try:
                if asset_id is None:
                    return
                self._mark_active_start()
                started = time.perf_counter()
                try:
                    await self._repository.begin_asset_enrichment(asset_ids=[asset_id])
                    outcome = await _run_enrichment_batch(
                        workspace_id=self._workspace_id,
                        asset_ids=[asset_id],
                        understanding_service=self._understanding_service,
                        embedding_service=self._embedding_service,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    outcome = _EnrichmentBatchResult(
                        errors=[
                            {
                                "asset_id": asset_id,
                                "stage": "enrichment",
                                "error": (str(exc) or type(exc).__name__)[:2000],
                            }
                        ],
                        stage_durations_ms={
                            PipelineStage.UNDERSTANDING.value: elapsed_ms,
                            PipelineStage.FEATURE_READY.value: 0.0,
                            PipelineStage.EMBEDDING.value: 0.0,
                            PipelineStage.INDEXING.value: 0.0,
                        },
                    )
                finally:
                    self._mark_active_end()
                self._outcomes.append(outcome)
            finally:
                self._queue.task_done()

    def _mark_active_start(self) -> None:
        if self._active_count == 0:
            self._active_started_at = time.perf_counter()
        self._active_count += 1

    def _mark_active_end(self) -> None:
        self._active_count -= 1
        if self._active_count == 0:
            self._active_elapsed_ms += (time.perf_counter() - self._active_started_at) * 1000

    def _normalized_stage_durations(self) -> dict[str, float]:
        raw = {
            stage.value: sum(
                outcome.stage_durations_ms.get(stage.value, 0.0) for outcome in self._outcomes
            )
            for stage in (
                PipelineStage.UNDERSTANDING,
                PipelineStage.FEATURE_READY,
                PipelineStage.EMBEDDING,
                PipelineStage.INDEXING,
            )
        }
        raw_total = sum(raw.values())
        if not raw_total:
            return raw
        return {
            stage: self._active_elapsed_ms * duration_ms / raw_total
            for stage, duration_ms in raw.items()
        }


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
        enrichment_pipeline: AssetEnrichmentPipeline | None = None
        try:
            if self._understanding_service is not None and self._embedding_service is not None:
                enrichment_pipeline = AssetEnrichmentPipeline(
                    settings=self._settings,
                    job_id=completion.job_id,
                    workspace_id=workspace_id,
                    repository=self._repository,
                    understanding_service=self._understanding_service,
                    embedding_service=self._embedding_service,
                )
                await enrichment_pipeline.start()
            if enrichment_pipeline is None:
                result = await self._runner.run(
                    completion.staged_path,
                    workspace_id,
                    job_id=completion.job_id,
                )
                return result
            result = await self._runner.run(
                completion.staged_path,
                workspace_id,
                job_id=completion.job_id,
                on_assets_stored=enrichment_pipeline.submit,
                finalize_job=False,
            )
            await enrichment_pipeline.submit(
                list(
                    getattr(
                        result,
                        "indexable_asset_ids",
                        getattr(result, "asset_ids", []),
                    )
                )
            )
            await enrichment_pipeline.finish()
            return result
        except Exception as exc:
            if enrichment_pipeline is not None:
                await enrichment_pipeline.abort()
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
