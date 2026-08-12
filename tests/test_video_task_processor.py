"""Unit contract for the Capsule video task processor's fenced callback."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capsule.enums import AssetType
from capsule.parsers.video import VideoAnalysisProgress
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_processor import CapsuleVideoTaskProcessor
from capsule.pipeline.video_task_runtime import (
    LeaseLostError,
    ProcessingTaskKind,
    ResourceClass,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
)
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile


class _AdaptiveParser:
    def __init__(self) -> None:
        self.sources: list[DiscoveredFile] = []

    async def assetize(self, source: DiscoveredFile, **_controls: Any) -> list[AssetDraft]:
        self.sources.append(source)
        progress = _controls.get("progress_callback")
        if progress is not None:
            await asyncio.to_thread(
                progress,
                VideoAnalysisProgress(
                    stage="decoding",
                    decoded_frames=7,
                    decoded_time_ms=1_200,
                    duration_ms=4_100,
                ),
            )
        # The parser result represents content-aware segmentation.  No processor
        # sample interval is supplied or interpreted here.
        return [
            AssetDraft(
                asset_type=AssetType.VIDEO_SEGMENT,
                file_name="demo.mp4",
                source_locator={"type": "time_range", "start_ms": 0, "end_ms": 1800},
                file_info={"representative_frames": [{"timestamp_ms": 200}]},
                transient_keyframe_jpegs=[b"adaptive-frame"],
            ),
            AssetDraft(
                asset_type=AssetType.VIDEO_SEGMENT,
                file_name="demo.mp4",
                source_locator={"type": "time_range", "start_ms": 1800, "end_ms": 4100},
                file_info={"representative_frames": [{"timestamp_ms": 2300}]},
                transient_keyframe_jpegs=[b"adaptive-frame"],
            ),
        ]


@pytest.mark.parametrize(
    "message_changes",
    [
        {"task_kind": ProcessingTaskKind.TEXT},
        {"resource_class": ResourceClass.CPU},
        {"route_key": "untrusted-route"},
        {"processor_version": 2},
    ],
)
def test_processor_rejects_each_message_lease_identity_mismatch(
    message_changes: dict[str, object],
) -> None:
    message = VideoTaskMessage(
        task_id="task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
    )
    lease = VideoTaskLease(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
        lease_token="claim-unique-token",
    )

    with pytest.raises(LeaseLostError, match="does not match"):
        CapsuleVideoTaskProcessor._assert_message_matches_lease(
            replace(message, **message_changes),  # type: ignore[arg-type]
            lease,
        )


class _Writer:
    async def persist(self, **values: Any) -> list[AssetCreate]:
        assets = values["assets"]
        validate = values["validate_asset_generation"]
        commit = values["on_asset_persisted"]
        progress = values.get("progress_callback")
        for asset in reversed(assets):
            await validate(asset)
            if progress is not None:
                await asyncio.to_thread(progress, asset, "rendering", 500, 1_000)
            await commit(asset, len(assets))
        return list(assets)


class _Committer:
    def __init__(self) -> None:
        self.validated: list[tuple[VideoTaskMessage, VideoTaskLease]] = []
        self.committed: list[tuple[VideoTaskMessage, VideoTaskLease, AssetCreate, int]] = []

    async def validate_lease_source(
        self, message: VideoTaskMessage, lease: VideoTaskLease
    ) -> bool:
        self.validated.append((message, lease))
        return True

    async def commit_segment(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        asset: AssetCreate,
        *,
        expected_asset_count: int,
    ) -> str:
        self.committed.append((message, lease, asset, expected_asset_count))
        return f"stored-{asset.asset_key[:8]}"


async def test_processor_reuses_adaptive_parser_and_fences_every_segment_commit(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"not-decoded-by-fake-parser")
    message = VideoTaskMessage(
        task_id="video-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=4,
        result_version=3,
        source_uri=video_path.as_uri(),
    )
    lease = VideoTaskLease(
        task_id="video-task-1",
        source_file_id="source-1",
        source_generation=4,
        attempt=2,
        worker_id="worker-1",
        result_version=3,
        lease_token="test-claim-token-1",
    )
    parser = _AdaptiveParser()
    committer = _Committer()
    progress: list[VideoTaskProgress] = []
    processor = CapsuleVideoTaskProcessor(
        parser=parser,  # type: ignore[arg-type]
        asset_factory=AssetFactory(),
        media_writer=_Writer(),  # type: ignore[arg-type]
        committer=committer,
        output_mode="materialized",
    )

    async def report(item: VideoTaskProgress) -> None:
        progress.append(item)

    result = await processor.process(message, lease, report)

    assert parser.sources[0].path == str(video_path)
    assert result.metadata["asset_count"] == 2
    assert result.metadata["source_generation"] == 4
    assert len(committer.committed) == 2
    assert all(call[0] == message and call[1] == lease for call in committer.committed)
    assert all(
        call[2].generation == 4 and call[2].source_file_id == "source-1"
        for call in committer.committed
    )
    assert all(call[3] == 2 for call in committer.committed)
    assert progress[0] == VideoTaskProgress(
        completed_units=1_200,
        total_units=4_100,
        detail={"stage": "decoding", "decoded_frames": 7},
    )
    assert any(
        item.detail.get("stage") == "rendering"
        and item.completed_units == 500
        and item.total_units == 1_000
        for item in progress
    )
    assert sorted(
        item.completed_units
        for item in progress
        if item.detail.get("stage") == "segment_committed"
    ) == [1, 2]


async def test_processor_rejects_a_lease_for_another_source_generation(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    message = VideoTaskMessage(
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=4,
        source_uri=video_path.as_uri(),
    )
    lease = VideoTaskLease(
        task_id=message.task_id or "",
        source_file_id="source-1",
        source_generation=5,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
        lease_token="test-claim-token-2",
    )
    processor = CapsuleVideoTaskProcessor(
        parser=_AdaptiveParser(),  # type: ignore[arg-type]
        asset_factory=AssetFactory(),
        media_writer=_Writer(),  # type: ignore[arg-type]
        committer=_Committer(),
    )

    try:
        await processor.process(message, lease, lambda _: _done())
    except LeaseLostError:
        return
    raise AssertionError("mismatched source generation must reject before parsing")


async def test_logical_processor_commits_timeline_metadata_without_media_writer(
    tmp_path: Path,
) -> None:
    class ForbiddenWriter:
        async def persist(self, **_values: Any) -> list[AssetCreate]:
            raise AssertionError("logical mode must not render or upload media")

    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    message = VideoTaskMessage(
        task_id="video-logical-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=4,
        source_uri=video_path.as_uri(),
    )
    lease = VideoTaskLease(
        task_id="video-logical-1",
        source_file_id="source-1",
        source_generation=4,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
        lease_token="test-claim-token-3",
    )
    committer = _Committer()
    processor = CapsuleVideoTaskProcessor(
        parser=_AdaptiveParser(),  # type: ignore[arg-type]
        asset_factory=AssetFactory(),
        media_writer=ForbiddenWriter(),  # type: ignore[arg-type]
        committer=committer,
        output_mode="logical",
    )

    result = await processor.process(message, lease, lambda _: _done())

    assert result.metadata["output_mode"] == "logical"
    assert len(committer.committed) == 2
    for _, _, asset, _ in committer.committed:
        assert asset.derived_file_uri is None
        assert asset.preview_uri is None
        assert asset.transient_keyframe_jpegs == []
        assert asset.file_info["video_output_mode"] == "logical"
        assert asset.processing_status.value == "pending"


async def _done() -> None:
    return None
