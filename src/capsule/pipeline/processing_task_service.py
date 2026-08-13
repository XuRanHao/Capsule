"""Operational wiring for durable CPU image and text processing tasks."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from capsule.config import Settings, get_settings
from capsule.db.processing_task_persistence import (
    CPU_IMAGE_ROUTE,
    CPU_TEXT_ROUTE,
    PostgresCpuProcessingTaskRepository,
    PostgresFencedAssetBatchCommitter,
    cpu_contract_for_extension,
)
from capsule.db.repositories import (
    AssetRepository,
    PreparedJobProcessingTask,
    PreparedProcessingTaskSubmission,
    ProcessingTaskSourceInput,
)
from capsule.db.session import Database
from capsule.model_clients.tokenization import LocalTokenCounter, TokenCounter
from capsule.parsers.discovery import sha256_file
from capsule.parsers.document import DocumentParser
from capsule.parsers.document_media import DocumentMediaExtractor, RapidOcrEngine
from capsule.pipeline.processing_task_processor import (
    DurableProcessingSource,
    ImageProcessingTaskProcessor,
    SourceContextsLoader,
    TextProcessingTaskProcessor,
)
from capsule.pipeline.runner import _collect_image_source_contexts, _processing_fingerprint
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskMessage,
    ProcessingTaskProcessor,
    ProcessingTaskQueue,
    RedisVideoTaskQueue,
    ResourceClass,
    RetryPolicy,
    VideoTaskRecoveryScheduler,
    VideoTaskRuntime,
)
from capsule.pipeline.video_task_service import VideoTaskScheduler, VideoTaskWorker
from capsule.schemas import DiscoveredFile, SourceContext
from capsule.video_sources import validate_video_source_root

logger = logging.getLogger(__name__)


class _SourceRepository(Protocol):
    async def create_cpu_processing_task_submission(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        result_version: int = 1,
        processor_version: int = 1,
    ) -> PreparedProcessingTaskSubmission: ...


class _TaskRepository(Protocol):
    async def mark_published(self, message: ProcessingTaskMessage) -> None: ...


@dataclass(frozen=True, slots=True)
class ProcessingTaskSubmission:
    job_id: str
    source_file_id: str
    generation: int
    task_id: str | None
    task_kind: ProcessingTaskKind
    route_key: str
    published: bool
    already_processed: bool = False


class ProcessingTaskSubmissionService:
    """Commit one CPU task to PostgreSQL before publishing its Redis delivery."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
        source_repository: _SourceRepository | None = None,
        queues: dict[ProcessingTaskKind, ProcessingTaskQueue] | None = None,
        task_repositories: dict[ProcessingTaskKind, _TaskRepository] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._source_repository = source_repository
        self._queues = queues or {}
        self._task_repositories = task_repositories or {}

    async def submit(
        self,
        *,
        input_path: Path,
        workspace_id: str,
        relative_path: str | None = None,
    ) -> ProcessingTaskSubmission:
        source_file = await asyncio.to_thread(
            _cpu_source,
            input_path,
            relative_path=relative_path,
        )
        contract = cpu_contract_for_extension(source_file.extension)
        validate_video_source_root(
            Path(source_file.path),
            import_root=self._settings.import_root,
            video_source_roots=self._settings.cpu_source_roots,
        )
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        source_repository = self._source_repository or AssetRepository(database)
        task_repository = self._task_repositories.get(contract.task_kind) or _task_repository(
            database,
            self._settings,
            contract.task_kind,
        )
        queue = self._queues.get(contract.task_kind) or _new_cpu_queue(
            self._settings,
            task_kind=contract.task_kind,
            consumer=f"submit-{_worker_identity()}",
        )
        owns_queue = contract.task_kind not in self._queues
        try:
            digest = await asyncio.to_thread(sha256_file, Path(source_file.path))
            submitted = await source_repository.create_cpu_processing_task_submission(
                workspace_id=workspace_id,
                input_path=Path(source_file.path),
                source_file=source_file,
                sha256=digest,
                mime_type=_mime_type(source_file),
                processing_fingerprint=_processing_fingerprint(
                    source_file,
                    self._settings,
                ),
            )
            if submitted.already_processed:
                return ProcessingTaskSubmission(
                    job_id=submitted.job_id,
                    source_file_id=submitted.source_file_id,
                    generation=submitted.generation,
                    task_id=None,
                    task_kind=contract.task_kind,
                    route_key=contract.route_key,
                    published=False,
                    already_processed=True,
                )
            if submitted.task_id is None:  # pragma: no cover - database invariant
                raise RuntimeError("queued CPU submission did not create a task id")
            message = ProcessingTaskMessage(
                task_id=submitted.task_id,
                job_id=submitted.job_id,
                workspace_id=workspace_id,
                source_file_id=submitted.source_file_id,
                generation=submitted.generation,
                source_uri=Path(source_file.path).as_uri(),
                task_kind=contract.task_kind,
                resource_class=contract.resource_class,
                route_key=contract.route_key,
                processor_version=contract.processor_version,
            )
            await _start_queue(queue)
            await queue.publish(message)
            await task_repository.mark_published(message)
            return ProcessingTaskSubmission(
                job_id=submitted.job_id,
                source_file_id=submitted.source_file_id,
                generation=submitted.generation,
                task_id=submitted.task_id,
                task_kind=contract.task_kind,
                route_key=contract.route_key,
                published=True,
            )
        finally:
            if owns_queue:
                await _close_queue(queue)
            if owns_database:
                await database.dispose()


class BrowserProcessingTaskSubmissionService:
    """Attach uploaded files to one existing browser import job."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
        source_repository: AssetRepository | None = None,
        queues: dict[ProcessingTaskKind, ProcessingTaskQueue] | None = None,
        task_repositories: dict[ProcessingTaskKind, Any] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._source_repository = source_repository
        self._queues = queues or {}
        self._task_repositories = task_repositories or {}

    async def submit_file(
        self,
        *,
        job_id: str,
        workspace_id: str,
        source_file: DiscoveredFile,
    ) -> ProcessingTaskSubmission:
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        repository = self._source_repository or AssetRepository(database)
        path = await asyncio.to_thread(lambda: Path(source_file.path).expanduser().resolve())
        validate_video_source_root(
            path,
            import_root=self._settings.import_root,
            video_source_roots=[
                *self._settings.video_source_roots,
                *self._settings.cpu_source_roots,
            ],
        )
        prepared: PreparedJobProcessingTask
        queue: ProcessingTaskQueue
        task_repository: Any
        try:
            digest = await asyncio.to_thread(sha256_file, path)
            prepared = await repository.create_job_processing_task(
                job_id=job_id,
                workspace_id=workspace_id,
                source_file=source_file,
                sha256=digest,
                mime_type=_mime_type(source_file),
                processing_fingerprint=_processing_fingerprint(source_file, self._settings),
            )
            task_kind = ProcessingTaskKind(prepared.task_kind)
            if prepared.already_processed:
                return ProcessingTaskSubmission(
                    job_id=prepared.job_id,
                    source_file_id=prepared.source_file_id,
                    generation=prepared.generation,
                    task_id=None,
                    task_kind=task_kind,
                    route_key=prepared.route_key,
                    published=False,
                    already_processed=True,
                )
            if prepared.task_id is None:  # pragma: no cover - database invariant
                raise RuntimeError("queued browser submission did not create a task id")
            if task_kind in {ProcessingTaskKind.VIDEO, ProcessingTaskKind.AUDIO}:
                from capsule.db.video_tasks import PostgresVideoTaskRepository
                from capsule.pipeline.video_task_service import _new_queue

                queue = _new_queue(
                    self._settings,
                    consumer=f"browser-submit-{_worker_identity()}",
                )
                task_repository = PostgresVideoTaskRepository(
                    database.session_factory,
                    task_kind=task_kind,
                    resource_class=ResourceClass.MPS_VIDEO,
                    route_key="mps_video",
                )
            else:
                queue = _new_cpu_queue(
                    self._settings,
                    task_kind=task_kind,
                    consumer=f"browser-submit-{_worker_identity()}",
                )
                task_repository = _task_repository(database, self._settings, task_kind)
            message = ProcessingTaskMessage(
                task_id=prepared.task_id,
                job_id=prepared.job_id,
                workspace_id=workspace_id,
                source_file_id=prepared.source_file_id,
                generation=prepared.generation,
                source_uri=path.as_uri(),
                task_kind=task_kind,
                resource_class=ResourceClass(prepared.resource_class),
                route_key=prepared.route_key,
                processor_version=prepared.processor_version,
            )
            await _start_queue(queue)
            try:
                await queue.publish(message)
                await task_repository.mark_published(message)
            finally:
                await _close_queue(queue)
            return ProcessingTaskSubmission(
                job_id=prepared.job_id,
                source_file_id=prepared.source_file_id,
                generation=prepared.generation,
                task_id=prepared.task_id,
                task_kind=task_kind,
                route_key=prepared.route_key,
                published=True,
            )
        finally:
            if owns_database:
                await database.dispose()

    async def submit_batch(
        self,
        *,
        job_id: str,
        workspace_id: str,
        source_files: list[DiscoveredFile],
        post_asset_action: str = "none",
    ) -> list[ProcessingTaskSubmission]:
        """Prepare the complete browser batch atomically, then publish by route."""
        if not source_files:
            raise ValueError("browser processing task batch cannot be empty")
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        repository = self._source_repository or AssetRepository(database)
        contexts_by_image = await asyncio.to_thread(
            _collect_image_source_contexts,
            source_files,
        )

        async def prepare_input(source: DiscoveredFile) -> ProcessingTaskSourceInput:
            path = await asyncio.to_thread(lambda: Path(source.path).expanduser().resolve())
            validate_video_source_root(
                path,
                import_root=self._settings.import_root,
                video_source_roots=[
                    *self._settings.video_source_roots,
                    *self._settings.cpu_source_roots,
                ],
            )
            digest = await asyncio.to_thread(sha256_file, path)
            return ProcessingTaskSourceInput(
                source_file=source,
                source_uri=path.as_uri(),
                sha256=digest,
                mime_type=_mime_type(source),
                processing_fingerprint=_processing_fingerprint(source, self._settings),
                source_contexts=tuple(
                    context.model_dump(mode="json")
                    for context in contexts_by_image.get(source.relative_path, [])
                ),
            )

        try:
            inputs = list(await asyncio.gather(*(prepare_input(item) for item in source_files)))
            prepared = await repository.prepare_import_task_batch(
                job_id=job_id,
                workspace_id=workspace_id,
                items=inputs,
                post_asset_action=post_asset_action,
            )
            queues = dict(self._queues)
            repositories = dict(self._task_repositories)
            owned_kinds: set[ProcessingTaskKind] = set()
            started_kinds: set[ProcessingTaskKind] = set()
            results: list[ProcessingTaskSubmission] = []
            try:
                for task in prepared:
                    if task.task_id is None:  # pragma: no cover - batch invariant
                        raise RuntimeError("browser batch task is missing its durable id")
                    task_kind = ProcessingTaskKind(task.task_kind)
                    if task_kind not in queues:
                        owned_kinds.add(task_kind)
                        if task_kind in {ProcessingTaskKind.VIDEO, ProcessingTaskKind.AUDIO}:
                            from capsule.db.video_tasks import PostgresVideoTaskRepository
                            from capsule.pipeline.video_task_service import _new_queue

                            queues[task_kind] = _new_queue(
                                self._settings,
                                consumer=f"browser-submit-{_worker_identity()}",
                            )
                            repositories[task_kind] = PostgresVideoTaskRepository(
                                database.session_factory,
                                task_kind=task_kind,
                                resource_class=ResourceClass.MPS_VIDEO,
                                route_key="mps_video",
                            )
                        else:
                            queues[task_kind] = _new_cpu_queue(
                                self._settings,
                                task_kind=task_kind,
                                consumer=f"browser-submit-{_worker_identity()}",
                            )
                            repositories[task_kind] = _task_repository(
                                database,
                                self._settings,
                                task_kind,
                            )
                    message = ProcessingTaskMessage(
                        task_id=task.task_id,
                        job_id=task.job_id,
                        workspace_id=workspace_id,
                        source_file_id=task.source_file_id,
                        generation=task.generation,
                        source_uri=task.source_uri,
                        task_kind=task_kind,
                        resource_class=ResourceClass(task.resource_class),
                        route_key=task.route_key,
                        processor_version=task.processor_version,
                    )
                    published = False
                    try:
                        if task_kind not in started_kinds:
                            await _start_queue(queues[task_kind])
                            started_kinds.add(task_kind)
                        await queues[task_kind].publish(message)
                        await repositories[task_kind].mark_published(message)
                        published = True
                    except Exception:
                        # PostgreSQL already owns the complete batch. The route
                        # scheduler will publish this queued task when Redis is
                        # available; do not turn a recoverable delivery outage
                        # into a failed browser Job.
                        logger.exception(
                            "browser task initial publish failed; scheduler will recover task=%s",
                            task.task_id,
                        )
                    results.append(
                        ProcessingTaskSubmission(
                            job_id=task.job_id,
                            source_file_id=task.source_file_id,
                            generation=task.generation,
                            task_id=task.task_id,
                            task_kind=task_kind,
                            route_key=task.route_key,
                            published=published,
                        )
                    )
            finally:
                await asyncio.gather(
                    *(_close_queue(queues[kind]) for kind in owned_kinds),
                    return_exceptions=True,
                )
            return results
        finally:
            if owns_database:
                await database.dispose()


class CpuProcessingTaskWorker(VideoTaskWorker):
    """A single-kind CPU worker using the generic durable runtime."""

    @classmethod
    def for_kind(
        cls,
        *,
        task_kind: ProcessingTaskKind,
        settings: Settings | None = None,
        worker_id: str | None = None,
        token_counter: TokenCounter | None = None,
        source_contexts_loader: SourceContextsLoader | None = None,
    ) -> CpuProcessingTaskWorker:
        if task_kind not in {ProcessingTaskKind.IMAGE, ProcessingTaskKind.TEXT}:
            raise ValueError("CPU worker task kind must be image or text")
        runtime_settings = settings or get_settings()
        database = Database(runtime_settings)
        identity = worker_id or _worker_identity()
        queue = _new_cpu_queue(
            runtime_settings,
            task_kind=task_kind,
            consumer=identity,
        )
        committer = PostgresFencedAssetBatchCommitter(database)
        loader = postgres_source_file_loader(database.session_factory)
        contexts_loader = source_contexts_loader or postgres_source_contexts_loader(
            database.session_factory
        )
        if task_kind is ProcessingTaskKind.IMAGE:
            processor: ProcessingTaskProcessor = ImageProcessingTaskProcessor(
                committer=committer,
                source_file_loader=loader,
                source_contexts_loader=contexts_loader,
            )
        else:
            processor = TextProcessingTaskProcessor(
                committer=committer,
                source_file_loader=loader,
                source_contexts_loader=contexts_loader,
                token_counter=token_counter
                or LocalTokenCounter(runtime_settings.document_tokenizer_path),
                document_parser=_document_parser(runtime_settings),
                min_tokens=runtime_settings.document_chunk_min_tokens,
                target_tokens=runtime_settings.document_chunk_target_tokens,
                max_tokens=runtime_settings.document_chunk_max_tokens,
                merge_max_tokens=runtime_settings.document_chunk_merge_max_tokens,
                parent_max_tokens=runtime_settings.document_parent_max_tokens,
            )
        route_key = _route_key(task_kind)
        runtime = VideoTaskRuntime(
            queue=queue,
            repository=_task_repository(database, runtime_settings, task_kind),
            processor=processor,
            worker_id=identity,
            retry_policy=RetryPolicy(
                max_attempts=runtime_settings.video_task_max_attempts,
                retry_delays_seconds=tuple(runtime_settings.video_task_retry_delays_seconds),
            ),
            heartbeat_seconds=runtime_settings.video_task_heartbeat_seconds,
            expected_route_key=route_key,
        )

        async def close() -> None:
            await _close_queue(queue)
            await database.dispose()

        concurrency = (
            runtime_settings.cpu_image_task_concurrency
            if task_kind is ProcessingTaskKind.IMAGE
            else runtime_settings.cpu_text_task_concurrency
        )
        return cls(
            runtime=runtime,
            queue=queue,
            close=close,
            concurrency=concurrency,
        )


class CpuProcessingTaskScheduler(VideoTaskScheduler):
    """Recover and redispatch one trusted CPU route from PostgreSQL facts."""

    @classmethod
    def for_kind(
        cls,
        *,
        task_kind: ProcessingTaskKind,
        settings: Settings | None = None,
    ) -> CpuProcessingTaskScheduler:
        if task_kind not in {ProcessingTaskKind.IMAGE, ProcessingTaskKind.TEXT}:
            raise ValueError("CPU scheduler task kind must be image or text")
        runtime_settings = settings or get_settings()
        database = Database(runtime_settings)
        queue = _new_cpu_queue(
            runtime_settings,
            task_kind=task_kind,
            consumer=f"scheduler-{_worker_identity()}",
        )

        async def close() -> None:
            await _close_queue(queue)
            await database.dispose()

        return cls(
            scheduler=VideoTaskRecoveryScheduler(
                queue=queue,
                repository=_task_repository(database, runtime_settings, task_kind),
            ),
            queue=queue,
            poll_seconds=runtime_settings.video_task_scheduler_poll_seconds,
            close=close,
        )


def postgres_source_file_loader(
    session_factory: Any,
) -> Callable[[ProcessingTaskMessage], Awaitable[DurableProcessingSource]]:
    """Load canonical source metadata from PostgreSQL, never from Redis fields."""
    from sqlalchemy import select

    from capsule.db.models import SourceFile

    async def load(message: ProcessingTaskMessage) -> DurableProcessingSource:
        async with session_factory() as session:
            source = await session.scalar(
                select(SourceFile).where(SourceFile.source_file_id == message.source_file_id)
            )
        if (
            source is None
            or source.workspace_id != message.workspace_id
            or source.processing_generation != message.generation
        ):
            raise ValueError("CPU processing task source is not the current canonical generation")
        path = _local_storage_path(source.storage_uri)
        return DurableProcessingSource(
            source_file=DiscoveredFile(
                path=str(path),
                relative_path=source.relative_path,
                extension=Path(source.relative_path).suffix.lower(),
                size_bytes=source.file_size_bytes,
            ),
            sha256=source.sha256,
        )

    return load


def postgres_source_contexts_loader(
    session_factory: Any,
) -> SourceContextsLoader:
    """Load trusted batch context from the durable task, never from Redis."""
    from sqlalchemy import select

    from capsule.db.video_tasks import VideoProcessingTask

    async def load(
        message: ProcessingTaskMessage,
        _source_file: DiscoveredFile,
    ) -> list[SourceContext]:
        async with session_factory() as session:
            payload = await session.scalar(
                select(VideoProcessingTask.input_payload).where(
                    VideoProcessingTask.task_id == message.task_id,
                    VideoProcessingTask.source_file_id == message.source_file_id,
                    VideoProcessingTask.source_generation == message.generation,
                )
            )
        raw_contexts = (payload or {}).get("source_contexts", [])
        return [SourceContext.model_validate(context) for context in raw_contexts]

    return load


def _task_repository(
    database: Database,
    settings: Settings,
    task_kind: ProcessingTaskKind,
) -> PostgresCpuProcessingTaskRepository:
    return PostgresCpuProcessingTaskRepository(
        database.session_factory,
        task_kind=task_kind,
        lease_seconds=settings.video_task_lease_seconds,
        progress_timeout_seconds=settings.video_task_progress_timeout_seconds,
        hard_timeout_seconds=settings.video_task_hard_timeout_seconds,
        redispatch_seconds=settings.video_task_redispatch_seconds,
        max_attempts=settings.video_task_max_attempts,
    )


def _new_cpu_queue(
    settings: Settings,
    *,
    task_kind: ProcessingTaskKind,
    consumer: str,
) -> RedisVideoTaskQueue:
    if task_kind is ProcessingTaskKind.IMAGE:
        stream = settings.cpu_image_task_stream
        group = settings.cpu_image_task_group
        dlq = settings.cpu_image_task_dlq_stream
    elif task_kind is ProcessingTaskKind.TEXT:
        stream = settings.cpu_text_task_stream
        group = settings.cpu_text_task_group
        dlq = settings.cpu_text_task_dlq_stream
    else:
        raise ValueError("CPU queue task kind must be image or text")
    return RedisVideoTaskQueue(
        redis_url=settings.redis_url,
        stream=stream,
        group=group,
        consumer=consumer,
        dlq_stream=dlq,
        claim_idle_ms=settings.video_task_claim_idle_ms,
    )


def _route_key(task_kind: ProcessingTaskKind) -> str:
    if task_kind is ProcessingTaskKind.IMAGE:
        return CPU_IMAGE_ROUTE
    if task_kind is ProcessingTaskKind.TEXT:
        return CPU_TEXT_ROUTE
    raise ValueError("CPU route task kind must be image or text")


def _document_parser(settings: Settings) -> DocumentParser:
    """Match PipelineRunner's document-media and local OCR construction exactly."""
    return DocumentParser(
        media_extractor=DocumentMediaExtractor(
            settings.document_media_root,
            ocr_engine=RapidOcrEngine() if settings.document_ocr_enabled else None,
            min_width=settings.document_ocr_min_edge,
            min_height=settings.document_ocr_min_edge,
            min_area=settings.document_ocr_min_area,
            min_ocr_confidence=settings.document_ocr_min_confidence,
        )
    )


def _cpu_source(
    input_path: Path,
    *,
    relative_path: str | None = None,
) -> DiscoveredFile:
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"processing task source does not exist: {path}")
    contract = cpu_contract_for_extension(path.suffix.lower())
    del contract
    canonical_relative = relative_path or path.name
    if Path(canonical_relative).is_absolute() or ".." in Path(canonical_relative).parts:
        raise ValueError("processing task relative_path must stay within its logical root")
    return DiscoveredFile(
        path=str(path),
        relative_path=Path(canonical_relative).as_posix(),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )


def _mime_type(source_file: DiscoveredFile) -> str:
    guessed, _ = mimetypes.guess_type(source_file.path)
    return guessed or "application/octet-stream"


def _local_storage_path(storage_uri: str) -> Path:
    from urllib.parse import unquote, urlparse

    parsed = urlparse(storage_uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError("CPU processing task source storage must be a local file URI")
    return Path(unquote(parsed.path) if parsed.scheme else storage_uri)


def _worker_identity() -> str:
    return f"{socket.gethostname()}-{__import__('os').getpid()}"


async def _start_queue(queue: ProcessingTaskQueue) -> None:
    start = getattr(queue, "start", None)
    if start is not None:
        await start()


async def _close_queue(queue: ProcessingTaskQueue) -> None:
    close = getattr(queue, "close", None)
    if close is not None:
        await close()
