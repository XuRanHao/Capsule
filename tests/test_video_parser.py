import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from capsule.parsers.video import (
    CandidateFrame,
    VideoParser,
    VideoSegmentationConfig,
    _candidate_frame,
    _iter_candidate_frame_groups,
    candidate_timestamps,
    filter_invalid_frames,
    merge_short_ranges,
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


def test_long_shot_uses_two_second_windows_after_ten_seconds() -> None:
    config = VideoSegmentationConfig()

    assert split_shot_windows(0, 10_000, config) == [(0, 10_000)]
    assert split_shot_windows(0, 11_000, config) == [
        (0, 2_000),
        (2_000, 4_000),
        (4_000, 6_000),
        (6_000, 8_000),
        (8_000, 10_000),
        (10_000, 11_000),
    ]


def test_sub_second_window_tail_merges_into_previous_window() -> None:
    config = VideoSegmentationConfig()

    assert split_shot_windows(0, 10_167, config) == [
        (0, 2_000),
        (2_000, 4_000),
        (4_000, 6_000),
        (6_000, 8_000),
        (8_000, 10_167),
    ]


def test_sub_second_scene_ranges_merge_with_temporal_neighbors() -> None:
    assert merge_short_ranges(
        [(0, 400), (400, 2_000), (2_000, 2_700), (2_700, 4_000)],
        minimum_ms=1_000,
    ) == [(0, 2_700), (2_700, 4_000)]

    assert merge_short_ranges([(0, 700)], minimum_ms=1_000) == [(0, 700)]


def test_candidate_sampling_uses_half_second_internal_points() -> None:
    assert candidate_timestamps(0, 3_000, 0.5, 20) == [
        500,
        1_000,
        1_500,
        2_000,
        2_500,
    ]
    assert candidate_timestamps(1_000, 3_000, 0.5, 20) == [1_500, 2_000, 2_500]
    assert candidate_timestamps(0, 400, 0.5, 20) == [200]
    sampled = candidate_timestamps(0, 10_000, 0.5, 20)
    assert len(sampled) == 19
    assert sampled[0] == 500
    assert sampled[-1] == 9_500


def test_candidate_frame_bounds_analysis_memory_after_full_resolution_metrics() -> None:
    original = np.full((216, 384, 3), 100, dtype=np.uint8)

    candidate = _candidate_frame(
        500,
        500,
        original,
        analysis_frame_max_edge=128,
    )

    assert original.shape == (216, 384, 3)
    assert candidate.frame.shape == (72, 128, 3)
    assert candidate.frame.flags.c_contiguous
    assert candidate.frame.nbytes < original.nbytes
    assert candidate.brightness == pytest.approx(100.0)


def test_candidate_frame_iterator_releases_completed_segment_groups() -> None:
    groups = _iter_candidate_frame_groups(
        VIDEO_FIXTURE,
        [[250, 500], [750, 1_000]],
        analysis_frame_max_edge=128,
    )

    first = next(groups)
    first_frame = weakref.ref(first[0].frame)
    del first
    second = next(groups)
    gc.collect()

    assert second
    assert first_frame() is None
    groups.close()


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
        assert asset.file_info["candidate_frame_count"] >= 1
        assert 1 <= len(asset.file_info["representative_frames"]) <= 3
