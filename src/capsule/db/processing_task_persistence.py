"""Trusted CPU image/text task contracts over the durable processing-task table.

Image and text work use separate trusted CPU routes. The durable ``task_kind``
remains the processor discriminator, while route separation lets schedulers
and consumer groups scale the two workloads independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from capsule.db.models import Asset, SourceFile
from capsule.db.repositories import _validate_asset_hierarchy
from capsule.db.video_asset_committer import (
    PostgresFencedVideoAssetCommitter,
    _live_lease_clauses,
    _message_matches_lease,
    _owned_live_lease_clauses,
    _upsert_asset,
)
from capsule.db.video_task_accounting import account_video_task_outcome
from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.enums import ProcessingStatus
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    ProcessingTaskKind,
    ResourceClass,
    VideoTaskLease,
    VideoTaskMessage,
)
from capsule.schemas import AssetCreate

CPU_IMAGE_ROUTE = "cpu_image"
CPU_TEXT_ROUTE = "cpu_text"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
TEXT_EXTENSIONS = frozenset({".md", ".txt", ".docx", ".pdf"})


@dataclass(frozen=True, slots=True)
class ProcessingTaskContract:
    task_kind: ProcessingTaskKind
    resource_class: ResourceClass
    route_key: str
    processor_version: int = 1


def cpu_contract_for_extension(extension: str) -> ProcessingTaskContract:
    """Map a supported source extension to its only trusted CPU contract."""
    suffix = extension.lower()
    if suffix in IMAGE_EXTENSIONS:
        kind = ProcessingTaskKind.IMAGE
        route_key = CPU_IMAGE_ROUTE
    elif suffix in TEXT_EXTENSIONS:
        kind = ProcessingTaskKind.TEXT
        route_key = CPU_TEXT_ROUTE
    else:
        raise ValueError(f"unsupported CPU processing source extension: {extension!r}")
    return ProcessingTaskContract(
        task_kind=kind,
        resource_class=ResourceClass.CPU,
        route_key=route_key,
    )


class PostgresCpuProcessingTaskRepository(PostgresVideoTaskRepository):
    """Fenced repository bound to one image or text CPU processor contract."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        task_kind: ProcessingTaskKind,
        processor_version: int = 1,
        **kwargs: object,
    ) -> None:
        if task_kind not in {ProcessingTaskKind.IMAGE, ProcessingTaskKind.TEXT}:
            raise ValueError("CPU task repository only supports image or text task kinds")
        route_key = (
            CPU_IMAGE_ROUTE if task_kind is ProcessingTaskKind.IMAGE else CPU_TEXT_ROUTE
        )
        super().__init__(
            session_factory,
            task_kind=task_kind.value,
            resource_class=ResourceClass.CPU.value,
            route_key=route_key,
            processor_version=processor_version,
            **kwargs,  # type: ignore[arg-type]
        )


class PostgresFencedProcessingAssetCommitter(PostgresFencedVideoAssetCommitter):
    """Neutral name for the generic fenced Asset commit transaction.

    The inherited implementation already fences every persisted task identity,
    including ``task_kind``, before publishing Assets or parent counters.
    """


class PostgresFencedAssetBatchCommitter:
    """Commit all CPU-produced Assets under one live, complete task lease.

    This is the commit boundary for image/text processors: Assets, current
    source publication, result-committed fact and parent accounting are one
    transaction. An exact replay returns the already committed Asset ids; any
    differing replay is rejected instead of replacing the durable result.
    """

    def __init__(self, database: Any) -> None:
        self._database = database

    async def validate_lease_source(
        self, message: VideoTaskMessage, lease: VideoTaskLease
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
            return bool(generation == message.generation)

    async def commit_assets(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        assets: list[AssetCreate],
        *,
        source_sha256: str,
    ) -> list[str]:
        """Persist a complete batch and make its source generation visible."""
        if not lease.lease_token:
            raise LeaseLostError("processing task lease is missing its claim token")
        if not _message_matches_lease(message, lease):
            raise LeaseLostError("processing task message does not match its database lease")
        if not assets:
            raise ValueError("processing task Asset batch cannot be empty")
        if not source_sha256:
            raise ValueError("processing task source_sha256 is required")
        if any(
            asset.workspace_id != message.workspace_id
            or asset.source_file_id != message.source_file_id
            or asset.generation != message.generation
            for asset in assets
        ):
            raise LeaseLostError("processing task Asset does not match its source generation")
        _validate_asset_hierarchy(assets, require_batch_parent=True)

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
                or source.sha256 != source_sha256
            ):
                raise LeaseLostError("source generation or content is no longer current")

            if task.status == "result_committed":
                return await _exact_replay_asset_ids(
                    session,
                    source_file_id=source.source_file_id,
                    generation=message.generation,
                    assets=assets,
                )

            stored_by_key: dict[str, str] = {}
            for asset in sorted(assets, key=_asset_commit_order):
                stored_by_key[asset.asset_key] = await _upsert_asset(
                    session,
                    source_file_id=source.source_file_id,
                    asset=asset,
                )
            await session.execute(
                delete(Asset).where(
                    Asset.source_file_id == source.source_file_id,
                    Asset.generation != message.generation,
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            accounted = await session.execute(
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
                            f"processing-task://{task.task_id}/generation/{task.source_generation}"
                        ),
                        "metadata": {
                            "source_file_id": task.source_file_id,
                            "source_generation": task.source_generation,
                            "asset_count": len(assets),
                        },
                    },
                    parent_accounted_at=func.now(),
                    updated_at=func.now(),
                )
            )
            if accounted.rowcount != 1:
                raise LeaseLostError("task lease expired before result commit")
            await account_video_task_outcome(
                session,
                parent_job_id=task.parent_job_id,
                outcome="completed",
            )
            return [stored_by_key[asset.asset_key] for asset in assets]


def _asset_commit_order(asset: AssetCreate) -> tuple[int, str]:
    """Ensure parent references exist before child Asset upserts."""
    return (1 if asset.parent_asset_key is not None else 0, asset.asset_key)


async def _exact_replay_asset_ids(
    session: Any,
    *,
    source_file_id: str,
    generation: int,
    assets: list[AssetCreate],
) -> list[str]:
    stored = {
        asset.asset_key: asset
        for asset in await session.scalars(
            select(Asset).where(
                Asset.source_file_id == source_file_id,
                Asset.generation == generation,
            )
        )
    }
    if len(stored) != len(assets) or any(
        current is None
        or current.asset_id != asset.asset_id
        or current.content_hash != asset.content_hash
        for asset in assets
        for current in (stored.get(asset.asset_key),)
    ):
        raise LeaseLostError("processing task result is already committed with different Assets")
    return [stored[asset.asset_key].asset_id for asset in assets]
