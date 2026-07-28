from pathlib import Path

import numpy as np
import pytest

from capsule.parsers.video import (
    CandidateFrame,
    VideoParser,
    VideoSegmentationConfig,
    candidate_timestamps,
    filter_invalid_frames,
    select_representative_frames,
    split_shot_windows,
)
from capsule.schemas import DiscoveredFile

VIDEO_FIXTURE = Path("data/dev-fixtures/nature/hiking-trip.mp4").resolve()
VIDEO_FIXTURE_SIZE = VIDEO_FIXTURE.stat().st_size


class FakeEmbedder:
    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [[float(index + 1), float((index % 3) + 1)] for index, _ in enumerate(frames)],
            dtype=np.float32,
        )


def _candidate(
    timestamp_ms: int,
    *,
    brightness: float = 120.0,
    contrast: float = 30.0,
    sharpness: float = 80.0,
    pixel: int = 80,
) -> CandidateFrame:
    return CandidateFrame(
        requested_ms=timestamp_ms,
        timestamp_ms=timestamp_ms,
        frame=np.full((48, 64, 3), pixel, dtype=np.uint8),
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
    )


def test_long_shot_uses_twenty_second_windows() -> None:
    config = VideoSegmentationConfig()

    assert split_shot_windows(0, 45_000, config) == [(0, 45_000)]
    assert split_shot_windows(0, 70_000, config) == [
        (0, 20_000),
        (20_000, 40_000),
        (40_000, 60_000),
        (60_000, 70_000),
    ]


def test_candidate_sampling_preserves_both_endpoints_and_cap() -> None:
    assert candidate_timestamps(0, 3_000, 5.0, 12) == [0, 3_000]
    assert candidate_timestamps(0, 20_000, 5.0, 12) == [0, 5_000, 10_000, 15_000, 20_000]
    sampled = candidate_timestamps(0, 100_000, 5.0, 12)
    assert len(sampled) == 12
    assert sampled[0] == 0
    assert sampled[-1] == 100_000


def test_invalid_frame_filter_rejects_blank_and_duplicate_frames() -> None:
    config = VideoSegmentationConfig()
    black = _candidate(0, brightness=0.0, contrast=0.0, sharpness=10.0, pixel=0)
    first = _candidate(5_000, pixel=80)
    duplicate = _candidate(10_000, pixel=80)
    distinct = _candidate(15_000, pixel=140)

    accepted = filter_invalid_frames([black, first, duplicate, distinct], config)

    assert [candidate.timestamp_ms for candidate in accepted] == [5_000, 15_000]
    assert black.invalid_reason == "black_frame"
    assert duplicate.invalid_reason == "near_duplicate"


def test_representatives_are_real_frames_in_time_order() -> None:
    config = VideoSegmentationConfig(cluster_silhouette_min=0.0)
    candidates = [_candidate(index * 1_000, pixel=20 + index * 30) for index in range(4)]
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]],
        dtype=np.float32,
    )

    selected = select_representative_frames(candidates, embeddings, config)

    assert 1 <= len(selected) <= 3
    assert selected == sorted(selected, key=lambda candidate: candidate.timestamp_ms)
    assert all(candidate in candidates for candidate in selected)


@pytest.mark.asyncio
async def test_video_fixture_becomes_logical_segment_assets() -> None:
    parser = VideoParser(concurrency=1, embedder=FakeEmbedder())
    source_file = DiscoveredFile(
        path=str(VIDEO_FIXTURE),
        relative_path="nature/hiking-trip.mp4",
        extension=".mp4",
        size_bytes=VIDEO_FIXTURE_SIZE,
    )

    assets = await parser.assetize(source_file)

    assert assets
    for index, asset in enumerate(assets):
        assert asset.asset_type.value == "video_segment"
        assert asset.raw_content is None
        assert asset.source_locator["segment_index"] == index
        assert asset.source_locator["type"] == "time_range"
        assert asset.source_locator["start_ms"] < asset.source_locator["end_ms"]
        assert asset.file_info["candidate_frame_count"] >= 2
        assert 1 <= len(asset.file_info["representative_frames"]) <= 3
