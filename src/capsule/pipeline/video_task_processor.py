"""Capsule's adaptive video pipeline behind the durable task-runtime protocol.

This module deliberately has no Redis or SQLAlchemy write logic.  The
``FencedVideoAssetCommitter`` boundary is where a PostgreSQL adapter must make
one transaction cover the task lease/source-generation fence, segment Asset
upsert/finalization, and the exactly-once parent-job counter update.  Keeping
that transaction at the database boundary prevents an upload worker from
accidentally committing an Asset under another task's lease.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from capsule.parsers.discovery import sha256_file
from capsule.parsers.video import (
    VideoAnalysisProgress,
    VideoCancellationToken,
    VideoParser,
)
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_media import VideoDerivedMediaWriter
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskResult,
)
from capsule.schemas import AssetCreate, DiscoveredFile
from capsule.video_output import VideoOutputMode, logical_video_asset


class FencedVideoAssetCommitter(Protocol):
    """Database boundary for one leased video's derived Segment Assets.

    ``commit_segment`` must check all of ``task_id``, ``attempt``, ``worker_id``
    and ``source_generation`` in its *same database transaction* as the Asset
    upsert/finalization and parent-job accounting.  It returns the durable Asset
    id or raises ``LeaseLostError`` when the caller has become stale.
    """

    async def validate_lease_source(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
    ) -> bool: ...

    async def commit_segment(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        asset: AssetCreate,
        *,
        expected_asset_count: int,
    ) -> str: ...


SourceFileLoader = Callable[[VideoTaskMessage], Awaitable[DiscoveredFile]]


class CapsuleVideoTaskProcessor:
    """Run adaptive ``VideoParser`` + media persistence for one leased task.

    The parser owns content-aware segmentation and MobileCLIP frame embedding;
    this processor never introduces a second fixed-interval frame sampler.
    ``VideoDerivedMediaWriter`` receives callbacks per ``persist`` invocation,
    so a shared writer cannot leak a different task's lease into a commit.  In
    ``logical`` mode it is never invoked: the fenced committer persists only
    content-aware ranges and representative-frame timestamps.
    """

    def __init__(
        self,
        *,
        parser: VideoParser,
        asset_factory: AssetFactory,
        media_writer: VideoDerivedMediaWriter | None,
        committer: FencedVideoAssetCommitter,
        source_file_loader: SourceFileLoader | None = None,
        output_mode: VideoOutputMode = "logical",
    ) -> None:
        self._parser = parser
        self._asset_factory = asset_factory
        self._media_writer = media_writer
        self._committer = committer
        self._source_file_loader = source_file_loader or _local_source_file
        self._output_mode = output_mode

    async def process(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        report_progress: Callable[[VideoTaskProgress], Awaitable[None]],
    ) -> VideoTaskResult:
        self._assert_message_matches_lease(message, lease)
        if not await self._committer.validate_lease_source(message, lease):
            raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")

        cancellation_token = VideoCancellationToken()
        loop = asyncio.get_running_loop()

        async def dispatch_progress(progress: VideoTaskProgress) -> None:
            await report_progress(progress)

        def report_from_worker(progress: VideoTaskProgress) -> None:
            future: Future[None] = asyncio.run_coroutine_threadsafe(
                dispatch_progress(progress),
                loop,
            )
            try:
                future.result()
            except BaseException:
                cancellation_token.cancel()
                raise

        def report_analysis(update: VideoAnalysisProgress) -> None:
            report_from_worker(
                VideoTaskProgress(
                    completed_units=update.decoded_time_ms,
                    total_units=update.duration_ms,
                    detail={
                        "stage": update.stage,
                        "decoded_frames": update.decoded_frames,
                    },
                )
            )

        try:
            source_file = await self._source_file_loader(message)
            source_sha256 = await asyncio.to_thread(sha256_file, Path(source_file.path))
            drafts = await self._parser.assetize(
                source_file,
                progress_callback=report_analysis,
                cancellation_token=cancellation_token,
            )
            if not drafts:
                raise ValueError("adaptive video parser produced no segments")
            assets = self._asset_factory.build_many(
                workspace_id=message.workspace_id,
                source_file_id=message.source_file_id,
                source_sha256=source_sha256,
                source_file=source_file,
                drafts=drafts,
                generation=message.generation,
            )
            if not assets:
                raise ValueError("adaptive video parser produced no Assets")
            if self._output_mode == "logical":
                assets = [logical_video_asset(asset) for asset in assets]

            committed_ids: list[str] = []
            committed_lock = asyncio.Lock()
            expected_asset_count = len(assets)
            await report_progress(
                VideoTaskProgress(
                    total_units=expected_asset_count,
                    detail={"stage": "segments_ready"},
                )
            )

            async def commit_segment(asset: AssetCreate, expected_count: int) -> str:
                if expected_count != expected_asset_count:
                    raise RuntimeError("video media manifest asset count changed during task")
                asset_id = await self._committer.commit_segment(
                    message,
                    lease,
                    asset,
                    expected_asset_count=expected_count,
                )
                async with committed_lock:
                    committed_ids.append(asset_id)
                    completed = len(committed_ids)
                await report_progress(
                    VideoTaskProgress(
                        completed_units=completed,
                        total_units=expected_asset_count,
                        detail={"stage": "segment_committed"},
                    )
                )
                return asset_id

            async def validate_asset(asset: AssetCreate) -> None:
                if (
                    asset.source_file_id != message.source_file_id
                    or asset.generation != message.generation
                ):
                    raise LeaseLostError("video writer attempted a different source generation")
                if not await self._committer.validate_lease_source(message, lease):
                    raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")

            def report_media(
                asset: AssetCreate,
                stage: str,
                completed_ms: int,
                total_ms: int,
            ) -> None:
                report_from_worker(
                    VideoTaskProgress(
                        completed_units=completed_ms,
                        total_units=total_ms,
                        detail={"stage": stage, "asset_key": asset.asset_key},
                    )
                )

            if self._output_mode == "logical":
                for asset in assets:
                    await validate_asset(asset)
                    await commit_segment(asset, expected_asset_count)
            else:
                if self._media_writer is None:
                    raise RuntimeError("video media writer is unavailable")
                await self._media_writer.persist(
                    source_file=source_file,
                    assets=assets,
                    on_asset_persisted=commit_segment,
                    validate_asset_generation=validate_asset,
                    progress_callback=report_media,
                    cancellation_token=cancellation_token,
                )
            if len(committed_ids) != expected_asset_count:
                raise RuntimeError(
                    "video task returned before all Segment Assets committed"
                )
            return VideoTaskResult(
                result_ref=f"video-task://{message.task_id}/generation/{message.generation}",
                metadata={
                    "source_file_id": message.source_file_id,
                    "source_generation": message.generation,
                    "asset_count": expected_asset_count,
                    "output_mode": self._output_mode,
                },
            )
        finally:
            cancellation_token.cancel()

    @staticmethod
    def _assert_message_matches_lease(message: VideoTaskMessage, lease: VideoTaskLease) -> None:
        if (
            message.task_id != lease.task_id
            or message.source_file_id != lease.source_file_id
            or message.generation != lease.source_generation
            or message.result_version != lease.result_version
        ):
            raise LeaseLostError("video task message does not match its database lease")


async def _local_source_file(message: VideoTaskMessage) -> DiscoveredFile:
    """Resolve the currently supported local-source task contract."""
    if not message.source_uri:
        raise ValueError("video task source_uri is required")
    parsed = urlparse(message.source_uri)
    if parsed.scheme not in ("", "file"):
        raise ValueError("video task source_uri must be a local path or file URI")
    raw_path = unquote(parsed.path) if parsed.scheme else message.source_uri
    path, size_bytes = await asyncio.to_thread(_resolve_existing_local_source, raw_path)
    return DiscoveredFile(
        path=str(path),
        relative_path=path.name,
        extension=path.suffix.lower(),
        size_bytes=size_bytes,
    )


def _resolve_existing_local_source(raw_path: str) -> tuple[Path, int]:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"video task source does not exist: {path}")
    return path, path.stat().st_size
