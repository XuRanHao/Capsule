"""PostgreSQL source of truth for reliable whole-video processing tasks.

Redis deliveries are intentionally not authoritative.  A worker may only mutate
its task while it owns the exact ``(task_id, attempt, owner_id,
source_generation)`` fence carried in :class:`VideoTaskLease`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from posixpath import normpath
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    bindparam,
    false,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from capsule.db.base import Base, id_factory
from capsule.db.models import ProcessingJob, SourceFile
from capsule.db.video_task_accounting import account_video_task_outcome
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ResourceClass,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskResult,
)


class VideoProcessingTask(Base):
    """Durable fact record for one source generation and result schema version."""

    __tablename__ = "video_processing_tasks"
    __table_args__ = (
        UniqueConstraint(
            "source_file_id",
            "source_generation",
            "result_version",
            name="uq_video_task_source_generation_result_version",
        ),
        Index("ix_video_processing_tasks_status_next_retry", "status", "next_retry_at"),
        Index(
            "ix_video_processing_tasks_kind_status_next_retry",
            "task_kind",
            "status",
            "next_retry_at",
        ),
        Index(
            "ix_video_processing_tasks_kind_identity",
            "source_file_id",
            "source_generation",
            "task_kind",
            "result_version",
        ),
        Index("ix_video_processing_tasks_parent_status", "parent_job_id", "status"),
    )

    task_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=id_factory("video_task")
    )
    parent_job_id: Mapped[str] = mapped_column(
        ForeignKey("processing_jobs.job_id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_file_id: Mapped[str] = mapped_column(
        ForeignKey("source_files.source_file_id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    result_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    task_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="video", index=True)
    resource_class: Mapped[str] = mapped_column(
        String(32), nullable=False, default="mps_video", index=True
    )
    processor_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    route_key: Mapped[str] = mapped_column(String(128), nullable=False, default="mps_video")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_id: Mapped[str | None] = mapped_column(String(255), index=True)
    lease_token: Mapped[str | None] = mapped_column(String(64))
    message_id: Mapped[str | None] = mapped_column(String(128))
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    lease_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    progress_deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    hard_deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    dlq_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    parent_accounted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PostgresVideoTaskRepository:
    """Async repository with ownership fencing for :mod:`video_task_runtime`.

    The fenced Asset committer accounts the parent in the transaction that makes
    the source generation visible.  ``complete`` then records the result and
    closes the task before the runtime acknowledges its Redis delivery.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        lease_seconds: float = 60.0,
        progress_timeout_seconds: float = 120.0,
        hard_timeout_seconds: float = 7_200.0,
        redispatch_seconds: float = 60.0,
        max_attempts: int = 4,
        task_kind: str = "video",
        resource_class: str = "mps_video",
        processor_version: int = 1,
        route_key: str = "mps_video",
    ) -> None:
        if (
            min(
                lease_seconds,
                progress_timeout_seconds,
                hard_timeout_seconds,
                redispatch_seconds,
            )
            <= 0
        ):
            raise ValueError("video task lease and deadline durations must be positive")
        if max_attempts < 1:
            raise ValueError("video task max_attempts must be positive")
        if not task_kind or not resource_class or not route_key:
            raise ValueError("video task kind, resource class, and route key are required")
        if processor_version < 1:
            raise ValueError("video task processor_version must be positive")
        self._session_factory = session_factory
        self._lease_duration = timedelta(seconds=lease_seconds)
        self._progress_timeout = timedelta(seconds=progress_timeout_seconds)
        self._hard_timeout = timedelta(seconds=hard_timeout_seconds)
        self._redispatch_duration = timedelta(seconds=redispatch_seconds)
        self._max_attempts = max_attempts
        self._task_kind = task_kind
        self._resource_class = resource_class
        self._processor_version = processor_version
        self._route_key = route_key

    async def create(
        self,
        message: VideoTaskMessage,
        *,
        hard_deadline_at: datetime | None = None,
    ) -> VideoProcessingTask:
        """Atomically create a task, or return its idempotent logical duplicate."""
        if not self._message_matches_identity(message):
            raise ValueError("video task message does not match repository identity")
        values: dict[str, Any] = {
            "task_id": message.task_id,
            "parent_job_id": message.job_id,
            "source_file_id": message.source_file_id,
            "source_generation": message.generation,
            "result_version": message.result_version,
            "task_kind": self._task_kind,
            "resource_class": self._resource_class,
            "processor_version": self._processor_version,
            "route_key": self._route_key,
            "status": "queued",
            "stage": "queued",
            "attempt": 0,
            "progress": {},
            "hard_deadline_at": hard_deadline_at,
        }
        stmt = (
            insert(VideoProcessingTask)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_video_task_source_generation_result_version")
            .returning(VideoProcessingTask)
        )
        async with self._session_factory() as session, session.begin():
            source = await session.scalar(
                select(SourceFile)
                .where(SourceFile.source_file_id == message.source_file_id)
                .with_for_update()
            )
            if source is None or source.processing_generation != message.generation:
                raise ValueError("video task source generation is no longer current")
            if source.workspace_id != message.workspace_id:
                raise ValueError("video task workspace does not match its source")
            parent_workspace_id = await session.scalar(
                select(ProcessingJob.workspace_id).where(ProcessingJob.job_id == message.job_id)
            )
            if parent_workspace_id != message.workspace_id:
                raise ValueError("video task workspace does not match its parent job")
            created = (await session.scalars(stmt)).one_or_none()
            if created is not None:
                return created
            existing = await session.scalar(
                select(VideoProcessingTask).where(
                    VideoProcessingTask.source_file_id == message.source_file_id,
                    VideoProcessingTask.source_generation == message.generation,
                    VideoProcessingTask.result_version == message.result_version,
                    *self._task_identity_clauses(),
                )
            )
            if existing is None:  # pragma: no cover - defensive against a broken schema
                raise RuntimeError("video task insert did not return a durable task")
            return existing

    async def claim_attempt(
        self,
        message: VideoTaskMessage,
        *,
        worker_id: str,
        receipt: str,
    ) -> VideoTaskLease | None:
        """Claim only the delivery which represents the next database attempt."""
        if not worker_id or not receipt:
            raise ValueError("worker_id and receipt are required")
        if not self._message_matches_identity(message):
            return None
        if message.task_id is None:  # pragma: no cover - normalized in message construction.
            return None
        previous_attempt = 0 if message.attempt == 0 else message.attempt - 1
        lease_token = uuid4().hex
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
                *self._task_identity_clauses(),
                VideoProcessingTask.attempt == previous_attempt,
                VideoProcessingTask.owner_id.is_(None),
                VideoProcessingTask.status.in_(("queued", "retry_wait")),
                or_(
                    VideoProcessingTask.status == "queued",
                    VideoProcessingTask.next_retry_at.is_(None),
                    VideoProcessingTask.next_retry_at <= func.now(),
                ),
            )
            .values(
                status="processing",
                stage="processing",
                attempt=VideoProcessingTask.attempt + 1,
                owner_id=worker_id,
                lease_token=lease_token,
                message_id=receipt,
                lease_deadline_at=func.now() + self._lease_duration,
                progress_deadline_at=func.now() + self._progress_timeout,
                hard_deadline_at=func.now() + self._hard_timeout,
                next_retry_at=None,
                updated_at=func.now(),
            )
        )
        async with self._session_factory() as session, session.begin():
            claim = await session.execute(stmt)
        if getattr(claim, "rowcount", 0) != 1:
            return None
        return VideoTaskLease(
            task_id=message.task_id,
            source_file_id=message.source_file_id,
            source_generation=message.generation,
            attempt=previous_attempt + 1,
            worker_id=worker_id,
            result_version=message.result_version,
            # The UPDATE predicate already fenced these four fields against
            # this repository's trusted processor contract. Reuse that exact
            # contract for the lease instead of trusting concurrently adapted
            # RETURNING labels for processor identity.
            task_kind=ProcessingTaskKind(self._task_kind),
            resource_class=ResourceClass(self._resource_class),
            route_key=self._route_key,
            processor_version=self._processor_version,
            lease_token=lease_token,
        )

    async def inspect_message_contract(self, message: VideoTaskMessage) -> bool | None:
        """Compare a transport message with its durable task contract.

        The lookup deliberately uses only ``task_id``: an incoming message must
        not be able to hide a mismatched task behind this repository's local
        routing configuration. ``None`` means the task does not exist.
        """
        stmt = (
            select(
                VideoProcessingTask,
                ProcessingJob.workspace_id,
                SourceFile.storage_uri,
            )
            .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
            .join(SourceFile, SourceFile.source_file_id == VideoProcessingTask.source_file_id)
            .where(VideoProcessingTask.task_id == message.task_id)
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).one_or_none()
        if row is None:
            return None
        task, parent_workspace_id, source_storage_uri = row
        return (
            task.parent_job_id == message.job_id
            and parent_workspace_id == message.workspace_id
            and task.source_file_id == message.source_file_id
            and task.source_generation == message.generation
            and task.result_version == message.result_version
            and task.task_kind == message.task_kind.value
            and message.resource_class is not None
            and task.resource_class == message.resource_class.value
            and task.route_key == message.route_key
            and task.processor_version == message.processor_version
            and _normalize_source_uri(message.source_uri)
            == _normalize_source_uri(source_storage_uri)
        )

    async def can_ack_unclaimed(self, message: VideoTaskMessage) -> bool:
        """ACK only terminal or result-committed duplicates, never active work."""
        if not self._message_matches_identity(message):
            return False
        stmt = select(
            VideoProcessingTask.status,
            VideoProcessingTask.dlq_published_at,
        ).where(
            VideoProcessingTask.task_id == message.task_id,
            VideoProcessingTask.source_file_id == message.source_file_id,
            VideoProcessingTask.source_generation == message.generation,
            VideoProcessingTask.result_version == message.result_version,
            *self._task_identity_clauses(),
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).one_or_none()
        if row is None:
            return False
        if row.status == "failed":
            return row.dlq_published_at is not None
        return row.status in {"result_committed", "completed", "invalidated", "cancelled"}

    async def heartbeat(self, lease: VideoTaskLease) -> bool:
        return await self._fenced_update(
            lease,
            statuses=("processing", "result_committed"),
            require_live_deadline=True,
            values={"lease_deadline_at": func.now() + self._lease_duration},
        )

    async def record_progress(self, lease: VideoTaskLease, progress: VideoTaskProgress) -> bool:
        payload: dict[str, Any] = {
            "completed_units": progress.completed_units,
            "total_units": progress.total_units,
            "detail": dict(progress.detail),
        }
        return await self._fenced_update(
            lease,
            statuses=("processing", "result_committed"),
            require_live_deadline=True,
            values={
                "progress": payload,
                "progress_deadline_at": func.now() + self._progress_timeout,
            },
        )

    async def complete(self, lease: VideoTaskLease, result: VideoTaskResult) -> bool:
        return await self._fenced_update(
            lease,
            statuses=("processing", "result_committed"),
            require_live_deadline=True,
            extra_clauses=(VideoProcessingTask.parent_accounted_at.is_not(None),),
            values={
                "status": "completed",
                "stage": "completed",
                "result": {"result_ref": result.result_ref, "metadata": dict(result.metadata)},
                "error_message": None,
                "owner_id": None,
                "lease_token": None,
                "message_id": None,
                "lease_deadline_at": None,
                "progress_deadline_at": None,
            },
        )

    async def mark_parent_accounted(self, lease: VideoTaskLease) -> bool:
        """Legacy explicit finalizer after a fenced asset commit has accounted parent."""
        return await self._fenced_update(
            lease,
            statuses=("processing",),
            require_live_deadline=True,
            extra_clauses=(VideoProcessingTask.parent_accounted_at.is_not(None),),
            values={
                "status": "completed",
                "stage": "completed",
                "owner_id": None,
                "lease_token": None,
                "message_id": None,
                "lease_deadline_at": None,
                "progress_deadline_at": None,
            },
        )

    async def schedule_retry(
        self,
        lease: VideoTaskLease,
        *,
        error: str,
        retry_at: float,
    ) -> bool:
        return await self._fenced_update(
            lease,
            statuses=("processing",),
            require_live_deadline=True,
            values={
                "status": "retry_wait",
                "stage": "retry_wait",
                "error_message": error[:2_000],
                "next_retry_at": _as_utc(retry_at),
                "owner_id": None,
                "lease_token": None,
                "message_id": None,
                "lease_deadline_at": None,
                "progress_deadline_at": None,
            },
        )

    async def fail(self, lease: VideoTaskLease, *, error: str) -> bool:
        """Fail a live task and account both its source and parent exactly once."""
        if not lease.lease_token:
            return False
        bounded_error = error[:2_000]
        clauses = [
            VideoProcessingTask.status == "processing",
            *self._fence_clauses(lease),
            VideoProcessingTask.parent_accounted_at.is_(None),
            or_(
                VideoProcessingTask.hard_deadline_at.is_(None),
                VideoProcessingTask.hard_deadline_at > func.now(),
            ),
            VideoProcessingTask.lease_deadline_at.is_not(None),
            VideoProcessingTask.lease_deadline_at > func.now(),
        ]
        failed = (
            update(VideoProcessingTask)
            .where(*clauses)
            .values(
                status="failed",
                stage="failed",
                error_message=bounded_error,
                parent_accounted_at=func.now(),
                owner_id=None,
                lease_token=None,
                message_id=None,
                lease_deadline_at=None,
                progress_deadline_at=None,
                next_retry_at=None,
                updated_at=func.now(),
            )
            .returning(
                VideoProcessingTask.parent_job_id,
                VideoProcessingTask.source_file_id,
                VideoProcessingTask.source_generation,
            )
        )
        async with self._session_factory() as session, session.begin():
            row = (await session.execute(failed)).one_or_none()
            if row is None:
                return False
            source = await session.get(SourceFile, row.source_file_id, with_for_update=True)
            if source is not None and source.processing_generation == row.source_generation:
                source.processing_status = "failed"
                source.error_message = bounded_error
            await account_video_task_outcome(
                session,
                parent_job_id=row.parent_job_id,
                outcome="failed",
            )
        return True

    async def invalidate(self, lease: VideoTaskLease, *, error: str = "source invalidated") -> bool:
        """Fence a stale source generation without allowing a worker takeover."""
        return await self._fenced_update(
            lease,
            statuses=("processing", "result_committed"),
            require_live_deadline=False,
            values={
                "status": "invalidated",
                "stage": "invalidated",
                "error_message": error[:2_000],
                "owner_id": None,
                "lease_token": None,
                "message_id": None,
                "lease_deadline_at": None,
                "progress_deadline_at": None,
                "next_retry_at": None,
            },
        )

    async def scan_queued_retry_finalizing(
        self, *, limit: int = 100, now: datetime | None = None
    ) -> list[VideoProcessingTask]:
        """Return work visible to a dispatcher or parent-accounting reconciler."""
        if limit < 1:
            raise ValueError("limit must be positive")
        current = now or datetime.now(UTC)
        stmt = (
            select(VideoProcessingTask)
            .where(
                *self._task_identity_clauses(),
                or_(
                    VideoProcessingTask.status == "queued",
                    and_(
                        VideoProcessingTask.status == "retry_wait",
                        or_(
                            VideoProcessingTask.next_retry_at.is_(None),
                            VideoProcessingTask.next_retry_at <= current,
                        ),
                    ),
                    VideoProcessingTask.status == "result_committed",
                ),
            )
            .order_by(VideoProcessingTask.created_at, VideoProcessingTask.task_id)
            .limit(limit)
        )
        async with self._session_factory() as session:
            return list((await session.scalars(stmt)).all())

    async def recover_timed_out(self, *, now: float) -> None:
        """Release expired processing leases and expose expired finalizers to repair."""
        current = _as_utc(now)
        stale = select(VideoProcessingTask).where(
            *self._task_identity_clauses(),
            VideoProcessingTask.status.in_(("processing", "result_committed")),
            VideoProcessingTask.owner_id.is_not(None),
            or_(
                and_(
                    VideoProcessingTask.lease_deadline_at.is_not(None),
                    VideoProcessingTask.lease_deadline_at <= current,
                ),
                and_(
                    VideoProcessingTask.progress_deadline_at.is_not(None),
                    VideoProcessingTask.progress_deadline_at <= current,
                ),
                and_(
                    VideoProcessingTask.hard_deadline_at.is_not(None),
                    VideoProcessingTask.hard_deadline_at <= current,
                ),
            ),
        )
        async with self._session_factory() as session, session.begin():
            tasks = list((await session.scalars(stale.with_for_update(skip_locked=True))).all())
            for task in tasks:
                # The observed owner/attempt/generation are repeated in the WHERE
                # clause, so a racing renewed lease cannot be reclaimed incorrectly.
                values: dict[str, Any] = {
                    "owner_id": None,
                    "lease_token": None,
                    "message_id": None,
                    "lease_deadline_at": None,
                    "progress_deadline_at": None,
                    "updated_at": func.now(),
                }
                if task.status == "processing":
                    if task.attempt >= self._max_attempts:
                        values.update(
                            status="failed",
                            stage="failed",
                            next_retry_at=None,
                            parent_accounted_at=func.now(),
                            error_message=task.error_message or "worker deadline expired",
                        )
                    else:
                        values.update(
                            status="retry_wait",
                            stage="retry_wait",
                            next_retry_at=current,
                            error_message=task.error_message or "worker lease expired",
                        )
                else:
                    values.update(status="completed", stage="completed")
                recovered = await session.scalar(
                    update(VideoProcessingTask)
                    .where(*self._fence_clauses(task))
                    .values(**values)
                    .returning(VideoProcessingTask.task_id)
                )
                if recovered is not None and recovered != task.task_id:  # pragma: no cover
                    raise RuntimeError("unexpected video task recovery fence result")
                if recovered is not None and values.get("status") == "failed":
                    source = await session.get(
                        SourceFile,
                        task.source_file_id,
                        with_for_update=True,
                    )
                    if (
                        source is not None
                        and source.processing_generation == task.source_generation
                    ):
                        source.processing_status = "failed"
                        source.error_message = str(values["error_message"])
                    await account_video_task_outcome(
                        session,
                        parent_job_id=task.parent_job_id,
                        outcome="failed",
                    )

    async def dispatchable_messages(self, *, now: float) -> list[VideoTaskMessage]:
        """Rebuild initial and retry deliveries from PostgreSQL durable state."""
        current = _as_utc(now)
        redispatch_before = current - self._redispatch_duration
        stmt = (
            select(VideoProcessingTask, ProcessingJob.workspace_id, SourceFile.storage_uri)
            .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
            .join(SourceFile, SourceFile.source_file_id == VideoProcessingTask.source_file_id)
            .where(
                *self._task_identity_clauses(),
                VideoProcessingTask.owner_id.is_(None),
                or_(
                    and_(
                        VideoProcessingTask.status == "queued",
                        or_(
                            VideoProcessingTask.last_published_at.is_(None),
                            VideoProcessingTask.last_published_at <= redispatch_before,
                        ),
                    ),
                    and_(
                        VideoProcessingTask.status == "retry_wait",
                        VideoProcessingTask.next_retry_at.is_not(None),
                        VideoProcessingTask.next_retry_at <= current,
                    ),
                ),
            )
            .order_by(VideoProcessingTask.created_at, VideoProcessingTask.task_id)
        )
        async with self._session_factory() as session:
            rows = list((await session.execute(stmt)).all())
        return [
            VideoTaskMessage(
                task_id=task.task_id,
                job_id=task.parent_job_id,
                workspace_id=workspace_id,
                source_file_id=task.source_file_id,
                generation=task.source_generation,
                attempt=(0 if task.attempt == 0 else task.attempt + 1),
                result_version=task.result_version,
                source_uri=storage_uri,
                task_kind=ProcessingTaskKind(task.task_kind),
                processor_version=task.processor_version,
                resource_class=ResourceClass(task.resource_class),
                route_key=task.route_key,
            )
            for task, workspace_id, storage_uri in rows
        ]

    async def mark_published(self, message: VideoTaskMessage) -> None:
        """Record publication after XADD; stale queued work remains redispatchable."""
        if not self._message_matches_identity(message):
            return
        previous_attempt = 0 if message.attempt == 0 else message.attempt - 1
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
                *self._task_identity_clauses(),
                VideoProcessingTask.attempt == previous_attempt,
                VideoProcessingTask.owner_id.is_(None),
                VideoProcessingTask.status.in_(("queued", "retry_wait")),
            )
            .values(
                status="queued",
                stage="queued",
                next_retry_at=None,
                last_published_at=func.now(),
                updated_at=func.now(),
            )
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    async def due_failed_dlq(self, *, now: float) -> list[tuple[VideoTaskMessage, str]]:
        """Return final failures whose durable DLQ event has not been published."""
        del now
        stmt = (
            select(
                VideoProcessingTask,
                ProcessingJob.workspace_id,
                SourceFile.storage_uri,
            )
            .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
            .join(SourceFile, SourceFile.source_file_id == VideoProcessingTask.source_file_id)
            .where(
                *self._task_identity_clauses(),
                VideoProcessingTask.status == "failed",
                VideoProcessingTask.dlq_published_at.is_(None),
            )
            .order_by(VideoProcessingTask.updated_at, VideoProcessingTask.task_id)
        )
        async with self._session_factory() as session:
            rows = list((await session.execute(stmt)).all())
        return [
            (
                VideoTaskMessage(
                    task_id=task.task_id,
                    job_id=task.parent_job_id,
                    workspace_id=workspace_id,
                    source_file_id=task.source_file_id,
                    generation=task.source_generation,
                    attempt=task.attempt,
                    result_version=task.result_version,
                    source_uri=storage_uri,
                    task_kind=ProcessingTaskKind(task.task_kind),
                    processor_version=task.processor_version,
                    resource_class=ResourceClass(task.resource_class),
                    route_key=task.route_key,
                ),
                task.error_message or "video task failed",
            )
            for task, workspace_id, storage_uri in rows
        ]

    async def mark_dlq_published(self, message: VideoTaskMessage) -> None:
        if not self._message_matches_identity(message):
            return
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
                *self._task_identity_clauses(),
                VideoProcessingTask.status == "failed",
                VideoProcessingTask.dlq_published_at.is_(None),
            )
            .values(dlq_published_at=func.now(), updated_at=func.now())
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    async def _fenced_update(
        self,
        lease: VideoTaskLease,
        *,
        statuses: tuple[str, ...],
        require_live_deadline: bool,
        extra_clauses: tuple[Any, ...] = (),
        values: dict[str, Any],
    ) -> bool:
        if not lease.lease_token:
            return False
        clauses = [
            VideoProcessingTask.status.in_(statuses),
            *self._fence_clauses(lease),
            *extra_clauses,
            or_(
                VideoProcessingTask.hard_deadline_at.is_(None),
                VideoProcessingTask.hard_deadline_at > func.now(),
            ),
        ]
        if require_live_deadline:
            clauses.extend(
                (
                    VideoProcessingTask.lease_deadline_at.is_not(None),
                    VideoProcessingTask.lease_deadline_at > func.now(),
                )
            )
        stmt = (
            update(VideoProcessingTask)
            .where(*clauses)
            .values(updated_at=func.now(), **values)
            .returning(VideoProcessingTask.task_id)
        )
        async with self._session_factory() as session, session.begin():
            updated_task_id = await session.scalar(stmt)
        return updated_task_id is not None

    def _fence_clauses(self, lease: VideoTaskLease | VideoProcessingTask) -> tuple[Any, ...]:
        """Return the full identity/ownership/generation fence for every write."""
        # A worker-provided lease without the per-claim token is never valid.
        # Recovery passes the persisted row instead, where a nullable token is
        # deliberately compared exactly (including ``IS NULL`` for old rows).
        if isinstance(lease, VideoTaskLease) and not lease.lease_token:
            token_clauses: tuple[Any, ...] = (false(),)
        elif lease.lease_token is None:
            token_clauses = (VideoProcessingTask.lease_token.is_(None),)
        else:
            token_clauses = (
                VideoProcessingTask.lease_token
                == bindparam(
                    "fence_lease_token",
                    lease.lease_token,
                    type_=String(64),
                ),
            )
        owner_id = lease.worker_id if isinstance(lease, VideoTaskLease) else lease.owner_id
        owner_clause = (
            VideoProcessingTask.owner_id.is_(None)
            if owner_id is None
            else VideoProcessingTask.owner_id
            == bindparam("fence_owner_id", owner_id, type_=String(255))
        )
        return (
            *self._lease_identity_clauses(lease),
            VideoProcessingTask.task_id
            == bindparam("fence_task_id", lease.task_id, type_=String(64)),
            VideoProcessingTask.source_file_id
            == bindparam(
                "fence_source_file_id",
                lease.source_file_id,
                type_=String(64),
            ),
            VideoProcessingTask.source_generation
            == bindparam(
                "fence_source_generation",
                lease.source_generation,
                type_=Integer(),
            ),
            VideoProcessingTask.result_version
            == bindparam(
                "fence_result_version",
                lease.result_version,
                type_=Integer(),
            ),
            VideoProcessingTask.attempt
            == bindparam("fence_attempt", lease.attempt, type_=Integer()),
            owner_clause,
            *token_clauses,
        )

    def _task_identity_clauses(self) -> tuple[Any, ...]:
        """Bind this repository to one complete persisted processor identity."""
        return (
            VideoProcessingTask.task_kind
            == bindparam("identity_task_kind", self._task_kind, type_=String(32)),
            VideoProcessingTask.resource_class
            == bindparam(
                "identity_resource_class",
                self._resource_class,
                type_=String(32),
            ),
            VideoProcessingTask.route_key
            == bindparam("identity_route_key", self._route_key, type_=String(128)),
            VideoProcessingTask.processor_version
            == bindparam(
                "identity_processor_version",
                self._processor_version,
                type_=Integer(),
            ),
        )

    @staticmethod
    def _lease_identity_clauses(lease: VideoTaskLease | VideoProcessingTask) -> tuple[Any, ...]:
        """Fence writes to the immutable identity carried by this exact lease."""
        if isinstance(lease, VideoTaskLease):
            task_kind = lease.task_kind.value
            resource_class = lease.resource_class.value
            route_key = lease.route_key
            processor_version = lease.processor_version
        else:
            task_kind = lease.task_kind
            resource_class = lease.resource_class
            route_key = lease.route_key
            processor_version = lease.processor_version
        return (
            VideoProcessingTask.task_kind
            == bindparam("fence_task_kind", task_kind, type_=String(32)),
            VideoProcessingTask.resource_class
            == bindparam("fence_resource_class", resource_class, type_=String(32)),
            VideoProcessingTask.route_key
            == bindparam("fence_route_key", route_key, type_=String(128)),
            VideoProcessingTask.processor_version
            == bindparam(
                "fence_processor_version",
                processor_version,
                type_=Integer(),
            ),
        )

    def _message_matches_identity(self, message: VideoTaskMessage) -> bool:
        """Reject transport identity that disagrees with this repository route.

        Route identity is transported as data but remains constrained by this
        server-owned repository configuration before database access.
        """
        route_key = getattr(message, "route_key", None)
        return (
            message.task_kind.value == self._task_kind
            and message.resource_class is not None
            and message.resource_class.value == self._resource_class
            and message.processor_version == self._processor_version
            and route_key == self._route_key
        )


def _as_utc(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _normalize_source_uri(value: str) -> str:
    """Canonicalize equivalent source URI spellings before contract comparison."""
    parsed = urlsplit(value)
    decoded_path = unquote(parsed.path)
    normalized_path = normpath(decoded_path) if decoded_path else ""
    if decoded_path.startswith("/") and not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            quote(normalized_path, safe="/%:@"),
            parsed.query,
            parsed.fragment,
        )
    )
