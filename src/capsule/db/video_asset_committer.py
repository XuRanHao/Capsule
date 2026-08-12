"""Atomic persistence boundary for a leased video's derived Segment Assets."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import delete, false, func, or_, select, update

from capsule.db.models import Asset, SourceFile
from capsule.db.repositories import (
    _asset_values,
    _resolve_asset_parent,
    _update_asset,
    _validate_asset_hierarchy,
)
from capsule.db.session import Database
from capsule.db.video_task_accounting import account_video_task_outcome
from capsule.db.video_tasks import VideoProcessingTask
from capsule.enums import AssetIndexRole, AssetNameSource, ProcessingStatus
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    VideoTaskLease,
    VideoTaskMessage,
)
from capsule.schemas import AssetCreate


class PostgresFencedVideoAssetCommitter:
    """Persist Segment Assets only while the exact PostgreSQL lease is live.

    Every segment write, source-generation publication, and the exactly-once
    parent-job success counter share one transaction.  This prevents a worker
    that loses its task lease between FFmpeg output and the database callback
    from making a stale generation visible.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def validate_lease_source(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
    ) -> bool:
        if not lease.lease_token or not _message_matches_lease(message, lease):
            return False
        async with self._database.session() as session:
            task = await session.scalar(
                select(VideoProcessingTask).where(*_live_lease_clauses(lease))
            )
            if task is None:
                return False
            generation = await session.scalar(
                select(SourceFile.processing_generation).where(
                    SourceFile.source_file_id == message.source_file_id
                )
            )
            return generation == message.generation

    async def commit_segment(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        asset: AssetCreate,
        *,
        expected_asset_count: int,
    ) -> str:
        """Fenced Asset upsert and generation publication in one transaction."""
        if not lease.lease_token:
            raise LeaseLostError("video task lease is missing its claim token")
        if expected_asset_count < 1:
            raise ValueError("expected_asset_count must be positive")
        if not _message_matches_lease(message, lease):
            raise LeaseLostError("video task message does not match its database lease")
        if (
            asset.workspace_id != message.workspace_id
            or asset.source_file_id != message.source_file_id
            or asset.generation != message.generation
        ):
            raise LeaseLostError("video Asset does not match its task source generation")
        _validate_asset_hierarchy([asset], require_batch_parent=False)

        async with self._database.session() as session, session.begin():
            task = await session.scalar(
                select(VideoProcessingTask)
                .where(*_owned_live_lease_clauses(lease))
                .with_for_update()
            )
            if task is None:
                raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")
            source = await session.get(SourceFile, message.source_file_id, with_for_update=True)
            if (
                source is None
                or source.workspace_id != message.workspace_id
                or source.processing_generation != message.generation
            ):
                raise LeaseLostError("source generation is no longer current")

            if task.status == "result_committed":
                committed = await session.scalar(
                    select(Asset).where(
                        Asset.source_file_id == source.source_file_id,
                        Asset.asset_key == asset.asset_key,
                        Asset.asset_id == asset.asset_id,
                        Asset.generation == message.generation,
                        Asset.content_hash == asset.content_hash,
                    )
                )
                if committed is None:
                    raise LeaseLostError(
                        "video task result is already committed with different Asset data"
                    )
                return committed.asset_id

            asset_id = await _upsert_asset(
                session,
                source_file_id=source.source_file_id,
                asset=asset,
            )
            stored_count = int(
                await session.scalar(
                    select(func.count(Asset.asset_id)).where(
                        Asset.source_file_id == source.source_file_id,
                        Asset.generation == message.generation,
                    )
                )
                or 0
            )
            if stored_count > expected_asset_count:
                raise ValueError("source generation contains more Assets than its upload manifest")
            if stored_count == expected_asset_count:
                await session.execute(
                    delete(Asset).where(
                        Asset.source_file_id == source.source_file_id,
                        Asset.generation != message.generation,
                    )
                )
                source.processing_status = ProcessingStatus.COMPLETED.value
                source.error_message = None
                await self._account_parent_once(
                    session,
                    task=task,
                    lease=lease,
                    asset_count=expected_asset_count,
                )
            return asset_id

    async def _account_parent_once(
        self,
        session: Any,
        *,
        task: VideoProcessingTask,
        lease: VideoTaskLease,
        asset_count: int,
    ) -> None:
        """Set the task accounting fact before incrementing its parent exactly once."""
        accounted = await session.scalar(
            update(VideoProcessingTask)
            .where(
                *_live_lease_clauses(lease),
                VideoProcessingTask.parent_accounted_at.is_(None),
            )
            .values(
                status="result_committed",
                stage="result_committed",
                result={
                    "result_ref": (
                        f"video-task://{task.task_id}/generation/{task.source_generation}"
                    ),
                    "metadata": {
                        "source_file_id": task.source_file_id,
                        "source_generation": task.source_generation,
                        "asset_count": asset_count,
                    },
                },
                parent_accounted_at=func.now(),
                updated_at=func.now(),
            )
            .returning(VideoProcessingTask.parent_job_id)
        )
        if accounted is None:
            return
        if accounted != task.parent_job_id:  # pragma: no cover - task_id is the primary key
            raise RuntimeError("video task parent changed during lease accounting")
        await account_video_task_outcome(
            session,
            parent_job_id=accounted,
            outcome="completed",
        )


async def _upsert_asset(
    session: Any,
    *,
    source_file_id: str,
    asset: AssetCreate,
) -> str:
    """Mirror AssetRepository.upsert_generated_asset inside the caller transaction."""
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
        .where(Asset.source_file_id == source_file_id, Asset.asset_key == asset.asset_key)
        .with_for_update()
    )
    if current is None:
        current = Asset(**_asset_values(asset))
        session.add(current)
    else:
        content_changed = current.content_hash != asset.content_hash
        user_name = (
            current.asset_name if current.asset_name_source == AssetNameSource.USER.value else None
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
    return cast(str, current.asset_id)


def _message_matches_lease(message: VideoTaskMessage, lease: VideoTaskLease) -> bool:
    return (
        message.task_id == lease.task_id
        and message.source_file_id == lease.source_file_id
        and message.generation == lease.source_generation
        and message.result_version == lease.result_version
        and message.task_kind == lease.task_kind
        and message.resource_class == lease.resource_class
        and message.route_key == lease.route_key
        and message.processor_version == lease.processor_version
    )


def _live_lease_clauses(lease: VideoTaskLease) -> tuple[Any, ...]:
    """The full immutable ownership fence required by every committing write."""
    return (
        *_owned_live_lease_clauses(lease),
        VideoProcessingTask.status == "processing",
    )


def _owned_live_lease_clauses(lease: VideoTaskLease) -> tuple[Any, ...]:
    """The live owner fence shared by writes and exact result replays."""
    token_clauses: tuple[Any, ...] = (
        (VideoProcessingTask.lease_token == lease.lease_token,)
        if lease.lease_token
        else (false(),)
    )
    return (
        VideoProcessingTask.task_id == lease.task_id,
        VideoProcessingTask.source_file_id == lease.source_file_id,
        VideoProcessingTask.source_generation == lease.source_generation,
        VideoProcessingTask.result_version == lease.result_version,
        VideoProcessingTask.task_kind == lease.task_kind.value,
        VideoProcessingTask.resource_class == lease.resource_class.value,
        VideoProcessingTask.route_key == lease.route_key,
        VideoProcessingTask.processor_version == lease.processor_version,
        VideoProcessingTask.attempt == lease.attempt,
        VideoProcessingTask.owner_id == lease.worker_id,
        *token_clauses,
        VideoProcessingTask.status.in_(("processing", "result_committed")),
        VideoProcessingTask.lease_deadline_at.is_not(None),
        VideoProcessingTask.lease_deadline_at > func.now(),
        VideoProcessingTask.progress_deadline_at.is_not(None),
        VideoProcessingTask.progress_deadline_at > func.now(),
        or_(
            VideoProcessingTask.hard_deadline_at.is_(None),
            VideoProcessingTask.hard_deadline_at > func.now(),
        ),
    )
