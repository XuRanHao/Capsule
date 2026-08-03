import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from capsule.parsers.video import (
    CandidateFrame,
    VideoMetadata,
    VideoParser,
    VideoSegmentationConfig,
    _candidate_frame,
    _content_atoms,
    _continuous_content_score,
    _merge_atoms,
    _merge_short_regions,
    _Region,
    _VideoAnalysis,
    filter_invalid_frames,
    select_representative_frames,
)
from capsule.schemas import DiscoveredFile

VIDEO_FIXTURE = Path("data/dev-fixtures/nature/hiking-trip.mp4").resolve()
VIDEO_FIXTURE_SIZE = VIDEO_FIXTURE.stat().st_size


class FakeEmbedder:
    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        values = []
        for frame in frames:
            mean = float(np.mean(frame)) / 255
            values.append([1.0, mean + 0.01])
        return np.asarray(values, dtype=np.float32)


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


def test_candidate_frame_is_cached_as_jpeg_without_another_video_decode() -> None:
    sampled = np.full((224, 224, 3), 100, dtype=np.uint8)

    candidate = _candidate_frame(
        500,
        500,
        sampled,
        jpeg_quality=85,
    )

    assert candidate.frame is not None
    assert candidate.frame.shape == (224, 224, 3)
    assert candidate.frame.flags.c_contiguous
    assert candidate.brightness == pytest.approx(100.0)
    assert candidate.jpeg_bytes is not None
    with Image.open(io.BytesIO(candidate.jpeg_bytes)) as image:
        assert image.size == (224, 224)


def test_continuous_content_score_matches_equal_weight_hsv_change() -> None:
    still = np.zeros((32, 48, 3), dtype=np.uint8)
    changed = still.copy()
    still[:, :, 2] = 40
    changed[:, :, 2] = 70

    assert _continuous_content_score(None, still) == 0.0
    assert _continuous_content_score(still, still) == 0.0
    # Only V changes by 30, so the equal-weight H/S/V score is 30 / 3.
    assert _continuous_content_score(still, changed) == 10.0


def test_video_config_requires_permanent_224px_keyframes() -> None:
    with pytest.raises(ValueError, match="exactly 224x224"):
        VideoSegmentationConfig(keyframe_size=256)


def test_video_parser_rejects_zero_concurrency_or_batch_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        VideoParser(concurrency=0, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="must be positive"):
        VideoParser(mobileclip_batch_size=0, embedder=FakeEmbedder())


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


def test_first_stage_distance_threshold_tracks_overall_video_change() -> None:
    config = VideoSegmentationConfig()
    candidates = [_candidate(index * 500) for index in range(4)]
    metadata = VideoMetadata(224, 224, 2_000, 30.0, "h264", "yuv420p", 1)
    quiet = np.asarray(
        [[1.0, 0.0], [0.999, 0.01], [0.998, 0.02], [0.997, 0.03]],
        dtype=np.float32,
    )
    active = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    quiet_analysis = _VideoAnalysis(candidates, quiet, np.zeros(60))
    active_analysis = _VideoAnalysis(candidates, active, np.zeros(60))

    _, quiet_threshold = _content_atoms(quiet_analysis, metadata, config)
    _, active_threshold = _content_atoms(active_analysis, metadata, config)

    assert quiet_threshold == pytest.approx(config.min_distance_threshold)
    assert active_threshold == pytest.approx(config.max_distance_threshold)


def test_second_stage_gate_follows_first_stage_and_duration_grows_logarithmically() -> None:
    config = VideoSegmentationConfig()
    analysis = _VideoAnalysis(
        [_candidate(0)],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.zeros(300),
    )
    atom = _Region((0,), (0,), 0, 10_000)
    short_metadata = VideoMetadata(224, 224, 10_000, 30.0, "h264", "yuv420p", 1)
    long_metadata = VideoMetadata(224, 224, 300_000, 30.0, "h264", "yuv420p", 1)

    _, short_info = _merge_atoms(
        [atom], analysis, short_metadata, config, distance_threshold=0.10
    )
    _, long_info = _merge_atoms(
        [atom], analysis, long_metadata, config, distance_threshold=0.25
    )

    assert short_info["similarity_gate"] == pytest.approx(0.85)
    assert long_info["similarity_gate"] == pytest.approx(0.70)
    assert short_info["duration_target"] == pytest.approx(10.0)
    assert short_info["duration_target"] < long_info["duration_target"] <= 20.0


def test_real_time_sub_second_region_merges_into_closest_neighbor() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]],
        dtype=np.float32,
    )
    regions = [
        _Region((0,), (0,), 0, 800),
        _Region((1,), (1,), 800, 2_000),
        _Region((2,), (2,), 2_000, 3_500),
    ]

    merged = _merge_short_regions(regions, embeddings, minimum_seconds=1.0)

    assert [(item.start_ms, item.end_ms) for item in merged] == [(0, 2_000), (2_000, 3_500)]


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
        assert asset.source_locator["source"] == "adaptive_content_two_stage"
        assert asset.source_locator["start_ms"] < asset.source_locator["end_ms"]
        assert asset.file_info["candidate_frame_count"] >= 1
        assert 1 <= len(asset.file_info["representative_frames"]) <= 3
        assert len(asset.transient_keyframe_jpegs) == len(
            asset.file_info["representative_frames"]
        )
        for payload in asset.transient_keyframe_jpegs:
            with Image.open(io.BytesIO(payload)) as image:
                assert image.size == (224, 224)
        segmentation = asset.file_info["segmentation"]
        assert 0.08 <= segmentation["first_stage_distance_threshold"] <= 0.25
        assert segmentation["second_stage_similarity_gate"] == pytest.approx(
            1 - segmentation["first_stage_distance_threshold"] - 0.05,
            abs=1e-6,
        )
