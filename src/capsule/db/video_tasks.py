"""PostgreSQL source of truth for reliable whole-video processing tasks.

Redis deliveries are intentionally not authoritative.  A worker may only mutate
its task while it owns the exact ``(task_id, attempt, owner_id,
source_generation)`` fence carried in :class:`VideoTaskLease`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
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
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    owner_id: Mapped[str | None] = mapped_column(String(255), index=True)
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
    ) -> None:
        if min(
            lease_seconds,
            progress_timeout_seconds,
            hard_timeout_seconds,
            redispatch_seconds,
        ) <= 0:
            raise ValueError("video task lease and deadline durations must be positive")
        if max_attempts < 1:
            raise ValueError("video task max_attempts must be positive")
        self._session_factory = session_factory
        self._lease_duration = timedelta(seconds=lease_seconds)
        self._progress_timeout = timedelta(seconds=progress_timeout_seconds)
        self._hard_timeout = timedelta(seconds=hard_timeout_seconds)
        self._redispatch_duration = timedelta(seconds=redispatch_seconds)
        self._max_attempts = max_attempts

    async def create(
        self,
        message: VideoTaskMessage,
        *,
        hard_deadline_at: datetime | None = None,
    ) -> VideoProcessingTask:
        """Atomically create a task, or return its idempotent logical duplicate."""
        values: dict[str, Any] = {
            "task_id": message.task_id,
            "parent_job_id": message.job_id,
            "source_file_id": message.source_file_id,
            "source_generation": message.generation,
            "result_version": message.result_version,
            "status": "queued",
            "stage": "queued",
            "attempt": 0,
            "progress": {},
            "hard_deadline_at": hard_deadline_at,
        }
        stmt = (
            insert(VideoProcessingTask)
            .values(**values)
            .on_conflict_do_nothing(
                constraint="uq_video_task_source_generation_result_version"
            )
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
        previous_attempt = 0 if message.attempt == 0 else message.attempt - 1
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
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
                message_id=receipt,
                lease_deadline_at=func.now() + self._lease_duration,
                progress_deadline_at=func.now() + self._progress_timeout,
                hard_deadline_at=func.now() + self._hard_timeout,
                next_retry_at=None,
                updated_at=func.now(),
            )
            .returning(
                VideoProcessingTask.task_id,
                VideoProcessingTask.source_file_id,
                VideoProcessingTask.source_generation,
                VideoProcessingTask.attempt,
                VideoProcessingTask.owner_id,
                VideoProcessingTask.result_version,
            )
        )
        async with self._session_factory() as session, session.begin():
            row = (await session.execute(stmt)).one_or_none()
        if row is None:
            return None
        return VideoTaskLease(
            task_id=row.task_id,
            source_file_id=row.source_file_id,
            source_generation=row.source_generation,
            attempt=row.attempt,
            worker_id=row.owner_id,
            result_version=row.result_version,
        )

    async def can_ack_unclaimed(self, message: VideoTaskMessage) -> bool:
        """ACK only terminal or result-committed duplicates, never active work."""
        stmt = select(
            VideoProcessingTask.status,
            VideoProcessingTask.dlq_published_at,
        ).where(
            VideoProcessingTask.task_id == message.task_id,
            VideoProcessingTask.source_file_id == message.source_file_id,
            VideoProcessingTask.source_generation == message.generation,
            VideoProcessingTask.result_version == message.result_version,
        )
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).one_or_none()
        if row is None:
            return True
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
                "message_id": None,
                "lease_deadline_at": None,
                "progress_deadline_at": None,
            },
        )

    async def fail(self, lease: VideoTaskLease, *, error: str) -> bool:
        """Fail a live task and account both its source and parent exactly once."""
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
                )
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
            )
            for task, workspace_id, storage_uri in rows
        ]

    async def mark_published(self, message: VideoTaskMessage) -> None:
        """Record publication after XADD; stale queued work remains redispatchable."""
        previous_attempt = 0 if message.attempt == 0 else message.attempt - 1
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
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

    async def due_failed_dlq(
        self, *, now: float
    ) -> list[tuple[VideoTaskMessage, str]]:
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
                ),
                task.error_message or "video task failed",
            )
            for task, workspace_id, storage_uri in rows
        ]

    async def mark_dlq_published(self, message: VideoTaskMessage) -> None:
        stmt = (
            update(VideoProcessingTask)
            .where(
                VideoProcessingTask.task_id == message.task_id,
                VideoProcessingTask.source_file_id == message.source_file_id,
                VideoProcessingTask.source_generation == message.generation,
                VideoProcessingTask.result_version == message.result_version,
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

    @staticmethod
    def _fence_clauses(lease: VideoTaskLease | VideoProcessingTask) -> tuple[Any, ...]:
        """Return the full identity/ownership/generation fence for every write."""
        return (
            VideoProcessingTask.task_id == lease.task_id,
            VideoProcessingTask.source_file_id == lease.source_file_id,
            VideoProcessingTask.source_generation == lease.source_generation,
            VideoProcessingTask.result_version == lease.result_version,
            VideoProcessingTask.attempt == lease.attempt,
            (
                VideoProcessingTask.owner_id == lease.worker_id
                if isinstance(lease, VideoTaskLease)
                else VideoProcessingTask.owner_id == lease.owner_id
            ),
        )


def _as_utc(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC)
