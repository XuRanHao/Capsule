"""Exactly-once parent-job accounting shared by fenced video task transitions."""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import func

from capsule.db.models import ProcessingJob
from capsule.enums import JobStatus, PipelineStage


async def account_video_task_outcome(
    session: Any,
    *,
    parent_job_id: str,
    outcome: Literal["completed", "failed"],
) -> None:
    """Increment one terminal count and close the parent job when all work ended.

    The caller invokes this only after its task row changed from
    ``parent_accounted_at IS NULL`` in the same transaction; that transition is
    the idempotency key for this otherwise simple counter update.
    """
    job = await session.get(ProcessingJob, parent_job_id, with_for_update=True)
    if job is None:  # pragma: no cover - foreign key protects this invariant
        raise RuntimeError("video task parent job disappeared during accounting")
    if outcome == "completed":
        job.completed_count += 1
    else:
        job.failed_count += 1
    if job.completed_count + job.failed_count < job.total_count:
        return
    if job.failed_count == 0:
        job.status = JobStatus.COMPLETED.value
        job.current_stage = PipelineStage.COMPLETED.value
    elif job.completed_count == 0:
        job.status = JobStatus.FAILED.value
        job.current_stage = PipelineStage.FAILED.value
    else:
        job.status = JobStatus.PARTIAL_FAILED.value
        job.current_stage = PipelineStage.COMPLETED.value
    job.completed_at = func.now()
