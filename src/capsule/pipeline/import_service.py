"""Browser folder import staging, with one retriable request per source file."""

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from uuid import uuid4

from fastapi import UploadFile
from pydantic import BaseModel, Field

from capsule.config import Settings
from capsule.db.repositories import AssetRepository
from capsule.enums import EmbeddingType, JobStatus, PipelineStage
from capsule.features import ACTIVE_EMBEDDING_TYPES
from capsule.parsers.discovery import SUPPORTED_EXTENSIONS, discover_files
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.processing_task_service import BrowserProcessingTaskSubmissionService
from capsule.pipeline.runner import PipelineRunner, PipelineRunResult
from capsule.pipeline.understanding import AssetUnderstandingService

logger = logging.getLogger(__name__)


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
    durable_dispatched: bool = False


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


class IncrementalClusterProcessor(Protocol):
    async def process_assets(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        asset_ids: list[str],
    ) -> object: ...


async def enrich_assets(
    *,
    job_id: str,
    workspace_id: str,
    asset_ids: list[str],
    repository: AssetRepository,
    understanding_service: AssetUnderstandingService,
    embedding_service: AssetEmbeddingService,
    incremental_cluster_processor: IncrementalClusterProcessor | None = None,
    force_understanding: bool = False,
    workflow_lease_token: str | None = None,
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
        incremental_cluster_processor=incremental_cluster_processor,
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
    if workflow_lease_token is None:
        await repository.finalize_enrichment(
            job_id=job_id,
            asset_ids=asset_ids,
            errors=batch.errors,
        )
    else:
        await repository.finalize_enrichment(
            job_id=job_id,
            asset_ids=asset_ids,
            errors=batch.errors,
            workflow_lease_token=workflow_lease_token,
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
    incremental_cluster_processor: IncrementalClusterProcessor | None = None,
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
        for embedding_type in ACTIVE_EMBEDDING_TYPES
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
    if incremental_cluster_processor is not None:
        await _process_incremental_clusters(
            processor=incremental_cluster_processor,
            workspace_id=workspace_id,
            embedding_types=[EmbeddingType(embedding.embedding_type) for embedding in embeddings],
            asset_ids=list(asset_ids),
        )
    return _EnrichmentBatchResult(
        errors=errors,
        stage_durations_ms=stage_durations_ms,
    )


async def _process_incremental_clusters(
    *,
    processor: IncrementalClusterProcessor,
    workspace_id: str,
    embedding_types: list[EmbeddingType],
    asset_ids: list[str],
) -> None:
    outcomes = await asyncio.gather(
        *(
            processor.process_assets(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                asset_ids=asset_ids,
            )
            for embedding_type in embedding_types
        ),
        return_exceptions=True,
    )
    for embedding_type, outcome in zip(embedding_types, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            logger.warning(
                "incremental clustering failed for workspace=%s embedding_type=%s: %s",
                workspace_id,
                embedding_type.value,
                str(outcome) or type(outcome).__name__,
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
        incremental_cluster_processor: IncrementalClusterProcessor | None = None,
    ) -> None:
        self._job_id = job_id
        self._workspace_id = workspace_id
        self._repository = repository
        self._understanding_service = understanding_service
        self._embedding_service = embedding_service
        self._incremental_cluster_processor = incremental_cluster_processor
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

        if self._asset_ids and self._incremental_cluster_processor is not None:
            await _process_incremental_clusters(
                processor=self._incremental_cluster_processor,
                workspace_id=self._workspace_id,
                embedding_types=list(ACTIVE_EMBEDDING_TYPES),
                asset_ids=list(self._asset_ids),
            )

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


class ImportWorkflowCoordinator:
    """Resume durable browser jobs after task-based assetization completes."""

    def __init__(
        self,
        *,
        repository: AssetRepository,
        understanding_service: AssetUnderstandingService,
        embedding_service: AssetEmbeddingService,
        incremental_cluster_processor: IncrementalClusterProcessor | None = None,
        worker_id: str | None = None,
        poll_seconds: float = 2.0,
        lease_seconds: float = 60.0,
    ) -> None:
        self._repository = repository
        self._understanding_service = understanding_service
        self._embedding_service = embedding_service
        self._incremental_cluster_processor = incremental_cluster_processor
        self._worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._closed = asyncio.Event()

    async def run_forever(self) -> None:
        while not self._closed.is_set():
            try:
                handled = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                handled = False
                logger.exception("durable import workflow coordinator iteration failed")
            if handled:
                continue
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass

    async def run_once(self) -> bool:
        claimed = await self._repository.claim_ready_import_workflow(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if claimed is None:
            return False
        job_id, workspace_id, lease_token = claimed
        failed = False
        enrichment: asyncio.Task[AssetEnrichmentResult] | None = None
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id=job_id, lease_token=lease_token)
        )
        try:
            asset_ids = await self._repository.list_import_job_asset_ids(job_id=job_id)
            await self._raise_if_heartbeat_ended(heartbeat)
            enrichment = asyncio.create_task(
                enrich_assets(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    asset_ids=asset_ids,
                    repository=self._repository,
                    understanding_service=self._understanding_service,
                    embedding_service=self._embedding_service,
                    incremental_cluster_processor=self._incremental_cluster_processor,
                    workflow_lease_token=lease_token,
                )
            )
            await self._await_enrichment_or_lease_loss(enrichment, heartbeat)
        except asyncio.CancelledError:
            await self._cancel_enrichment(enrichment)
            await self._release_lease_after_cancellation(
                job_id=job_id,
                lease_token=lease_token,
            )
            raise
        except Exception as exc:
            failed = True
            await self._repository.release_import_workflow(
                job_id=job_id,
                worker_id=self._worker_id,
                lease_token=lease_token,
                error=str(exc) or type(exc).__name__,
            )
            logger.exception("durable import enrichment failed for job=%s", job_id)
        finally:
            await self._cancel_enrichment(enrichment)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        return not failed

    @staticmethod
    async def _cancel_enrichment(
        enrichment: asyncio.Task[AssetEnrichmentResult] | None,
    ) -> None:
        if enrichment is None:
            return
        if not enrichment.done():
            enrichment.cancel()
        await asyncio.gather(enrichment, return_exceptions=True)

    async def _release_lease_after_cancellation(
        self,
        *,
        job_id: str,
        lease_token: str,
    ) -> None:
        """Release the lease before the lifespan can dispose the database."""
        release = asyncio.create_task(
            self._repository.release_import_workflow(
                job_id=job_id,
                worker_id=self._worker_id,
                lease_token=lease_token,
                error="import workflow coordinator cancelled",
            )
        )
        while not release.done():
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                # Preserve cleanup across repeated cancellation; the caller
                # re-raises its original cancellation once this returns.
                pass
            except Exception:
                break
        if release.cancelled():
            return
        try:
            release.result()
        except Exception:
            logger.exception(
                "failed to release durable import workflow lease after cancellation job=%s",
                job_id,
            )

    async def _raise_if_heartbeat_ended(self, heartbeat: asyncio.Task[None]) -> None:
        """Ensure the first lease renewal succeeded before enriching any assets."""
        await asyncio.sleep(0)
        if heartbeat.done():
            await heartbeat

    async def _await_enrichment_or_lease_loss(
        self,
        enrichment: asyncio.Task[AssetEnrichmentResult],
        heartbeat: asyncio.Task[None],
    ) -> AssetEnrichmentResult:
        """Fail closed if the durable lease ends before enrichment is terminal."""
        done, _ = await asyncio.wait(
            {enrichment, heartbeat},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat in done:
            try:
                await heartbeat
            except BaseException:
                enrichment.cancel()
                await asyncio.gather(enrichment, return_exceptions=True)
                raise
            raise RuntimeError("durable import workflow heartbeat stopped unexpectedly")
        return await enrichment

    async def close(self) -> None:
        self._closed.set()

    async def _heartbeat(self, *, job_id: str, lease_token: str) -> None:
        while True:
            renewed = await self._repository.heartbeat_import_workflow(
                job_id=job_id,
                worker_id=self._worker_id,
                lease_token=lease_token,
                lease_seconds=self._lease_seconds,
            )
            if not renewed:
                raise RuntimeError("durable import workflow lease was lost")
            await asyncio.sleep(self._lease_seconds / 3)


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
        incremental_cluster_processor: IncrementalClusterProcessor | None = None,
        durable_task_submitter: BrowserProcessingTaskSubmissionService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._runner = runner
        self._understanding_service = understanding_service
        self._embedding_service = embedding_service
        self._incremental_cluster_processor = incremental_cluster_processor
        self._durable_task_submitter = durable_task_submitter
        self._active_executions: dict[str, tuple[str, asyncio.Task[Any]]] = {}

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
            size_bytes = await asyncio.to_thread(
                _copy_upload_atomically,
                file.file,
                target,
                self._settings.import_file_max_bytes,
            )
            await self._repository.mark_import_upload_activity(
                job_id=job_id,
                workspace_id=workspace_id,
            )
            return size_bytes
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
        source_files = await asyncio.to_thread(discover_files, staged_path)
        file_count = len(source_files)
        if file_count == 0:
            raise ImportSubmissionError(
                "at least one supported file must be uploaded before completion"
            )
        post_asset_action = (
            "enrich"
            if self._understanding_service is not None
            and self._embedding_service is not None
            else "none"
        )
        if self._durable_task_submitter is not None:
            await self._durable_task_submitter.submit_batch(
                job_id=job_id,
                workspace_id=workspace_id,
                source_files=source_files,
                post_asset_action=post_asset_action,
            )
        else:
            await self._repository.start_import_job(
                job_id=job_id,
                total_count=file_count,
                post_asset_action=post_asset_action,
            )
        return ImportCompletion(
            job_id=job_id,
            staged_path=staged_path,
            file_count=file_count,
            durable_dispatched=self._durable_task_submitter is not None,
        )

    async def execute(
        self,
        *,
        completion: ImportCompletion,
        workspace_id: str,
    ) -> PipelineRunResult | None:
        enrichment_pipeline: AssetEnrichmentPipeline | None = None
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_executions[completion.job_id] = (workspace_id, current_task)
        try:
            if completion.durable_dispatched:
                return PipelineRunResult(
                    job_id=completion.job_id,
                    workspace_id=workspace_id,
                    file_count=completion.file_count,
                    succeeded_count=0,
                    failed_count=0,
                    asset_count=0,
                )
            if self._understanding_service is not None and self._embedding_service is not None:
                enrichment_pipeline = AssetEnrichmentPipeline(
                    settings=self._settings,
                    job_id=completion.job_id,
                    workspace_id=workspace_id,
                    repository=self._repository,
                    understanding_service=self._understanding_service,
                    embedding_service=self._embedding_service,
                    incremental_cluster_processor=self._incremental_cluster_processor,
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
        except asyncio.CancelledError:
            if enrichment_pipeline is not None:
                await enrichment_pipeline.abort()
            return None
        except Exception as exc:
            if enrichment_pipeline is not None:
                await enrichment_pipeline.abort()
            message = str(exc) or type(exc).__name__
            await self._repository.fail_job(job_id=completion.job_id, error=message)
            return None
        finally:
            active = self._active_executions.get(completion.job_id)
            if active is not None and active[1] is current_task:
                self._active_executions.pop(completion.job_id, None)

    async def cancel_active_jobs(self, *, workspace_id: str) -> int:
        tasks = [
            task
            for active_workspace_id, task in self._active_executions.values()
            if active_workspace_id == workspace_id and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

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
