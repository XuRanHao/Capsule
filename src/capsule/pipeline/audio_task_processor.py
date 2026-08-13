"""Audio parser adapter for Capsule's durable shared MPS task runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Future

from capsule.parsers.audio import AudioParser
from capsule.parsers.discovery import sha256_file
from capsule.parsers.video import VideoCancellationToken
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_processor import FencedVideoAssetCommitter, _local_source_file
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    ProcessingTaskKind,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskResult,
)


class CapsuleAudioTaskProcessor:
    def __init__(
        self,
        *,
        parser: AudioParser,
        asset_factory: AssetFactory,
        committer: FencedVideoAssetCommitter,
    ) -> None:
        self._parser = parser
        self._asset_factory = asset_factory
        self._committer = committer

    async def process(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        report_progress: Callable[[VideoTaskProgress], Awaitable[None]],
    ) -> VideoTaskResult:
        if message.task_kind is not ProcessingTaskKind.AUDIO:
            raise ValueError("audio processor received a non-audio task")
        if not await self._committer.validate_lease_source(message, lease):
            raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")
        token = VideoCancellationToken()
        loop = asyncio.get_running_loop()

        async def dispatch_progress(completed_ms: int, total_ms: int) -> None:
            await report_progress(
                VideoTaskProgress(
                    completed_units=completed_ms,
                    total_units=total_ms,
                    detail={"stage": "audio_embedding"},
                )
            )

        def progress(completed_ms: int, total_ms: int) -> None:
            future: Future[None] = asyncio.run_coroutine_threadsafe(
                dispatch_progress(completed_ms, total_ms),
                loop,
            )
            try:
                future.result()
            except BaseException:
                token.cancel()
                raise

        try:
            source = await _local_source_file(message)
            digest = await asyncio.to_thread(sha256_file, __import__("pathlib").Path(source.path))
            drafts = await self._parser.assetize(
                source,
                progress_callback=progress,
                cancellation_token=token,
            )
            assets = self._asset_factory.build_many(
                workspace_id=message.workspace_id,
                source_file_id=message.source_file_id,
                source_sha256=digest,
                source_file=source,
                drafts=drafts,
                generation=message.generation,
            )
            if not assets:
                raise ValueError("audio parser produced no Assets")
            ids: list[str] = []
            for asset in assets:
                if not await self._committer.validate_lease_source(message, lease):
                    raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")
                ids.append(
                    await self._committer.commit_segment(
                        message,
                        lease,
                        asset,
                        expected_asset_count=len(assets),
                    )
                )
                await report_progress(
                    VideoTaskProgress(
                        completed_units=len(ids),
                        total_units=len(assets),
                        detail={"stage": "segment_committed"},
                    )
                )
            return VideoTaskResult(
                result_ref=f"audio-task://{message.task_id}/generation/{message.generation}",
                metadata={
                    "source_file_id": message.source_file_id,
                    "source_generation": message.generation,
                    "asset_count": len(ids),
                    "output_mode": "logical",
                },
            )
        finally:
            token.cancel()
