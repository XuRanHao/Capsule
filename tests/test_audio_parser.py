from pathlib import Path

import numpy as np
import pytest

from capsule.enums import AssetType
from capsule.parsers.audio import AudioParser, AudioSegmentationConfig
from capsule.parsers.temporal_clustering import cluster_contiguous_windows
from capsule.schemas import DiscoveredFile


def test_contiguous_clustering_merges_only_short_adjacent_regions() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] + [[-1.0, 0.0]] * 4,
        dtype=np.float32,
    )
    regions, _ = cluster_contiguous_windows(
        embeddings,
        sample_timestamps_ms=[500 + index * 1000 for index in range(9)],
        duration_ms=9000,
        minimum_segment_seconds=3.0,
        distance_quantile=0.5,
        minimum_distance_threshold=0.1,
        maximum_distance_threshold=0.1,
    )

    assert regions[0].start_ms == 0
    assert regions[-1].end_ms == 9000
    assert all(region.duration_seconds >= 3 for region in regions)
    assert all(
        left.end_ms == right.start_ms
        for left, right in zip(regions, regions[1:], strict=False)
    )


def test_contiguous_clustering_has_no_maximum_duration() -> None:
    embeddings = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (240, 1))
    regions, _ = cluster_contiguous_windows(
        embeddings,
        sample_timestamps_ms=[500 + index * 1000 for index in range(240)],
        duration_ms=240_000,
        minimum_segment_seconds=3.0,
        distance_quantile=0.75,
        minimum_distance_threshold=0.08,
        maximum_distance_threshold=0.25,
    )

    assert [(item.start_ms, item.end_ms) for item in regions] == [(0, 240_000)]


def test_contiguous_clustering_accepts_silent_zero_vectors() -> None:
    regions, _ = cluster_contiguous_windows(
        np.zeros((8, 4), dtype=np.float32),
        sample_timestamps_ms=[500 + index * 1000 for index in range(8)],
        duration_ms=8000,
        minimum_segment_seconds=3.0,
        distance_quantile=0.75,
        minimum_distance_threshold=0.08,
        maximum_distance_threshold=0.25,
    )

    assert [(item.start_ms, item.end_ms) for item in regions] == [(0, 8000)]


@pytest.mark.asyncio
async def test_audio_parser_uses_independent_one_second_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"fixture")
    windows_seen: list[np.ndarray] = []

    class Embedder:
        def embed(self, waveforms: list[np.ndarray], *, sample_rate: int) -> np.ndarray:
            assert sample_rate == 32_000
            windows_seen.extend(waveforms)
            return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(waveforms), 1))

    monkeypatch.setattr(
        "capsule.parsers.audio._probe_audio",
        lambda _source, _token: __import__(
            "capsule.parsers.audio", fromlist=["AudioMetadata"]
        ).AudioMetadata(7500, 44_100, 2, "pcm_s16le", 7),
    )

    def fake_decode(
        _source: Path,
        *,
        metadata: object,
        embedder: object,
        config: AudioSegmentationConfig,
        progress_callback: object,
        cancellation_token: object,
    ) -> tuple[np.ndarray, list[int]]:
        del metadata, progress_callback, cancellation_token
        chunks = [np.zeros(32_000, dtype=np.float32) for _ in range(7)]
        chunks.append(np.zeros(32_000, dtype=np.float32))
        values = embedder.embed(chunks, sample_rate=config.sample_rate)  # type: ignore[attr-defined]
        return values, [500, 1500, 2500, 3500, 4500, 5500, 6500, 7250]

    monkeypatch.setattr("capsule.parsers.audio._decode_and_embed", fake_decode)
    parser = AudioParser(embedder=Embedder(), config=AudioSegmentationConfig())
    drafts = await parser.assetize(
        DiscoveredFile(
            path=str(source),
            relative_path=source.name,
            extension=".wav",
            size_bytes=source.stat().st_size,
        )
    )

    assert len(windows_seen) == 8
    assert all(len(window) == 32_000 for window in windows_seen)
    assert drafts[0].asset_type is AssetType.AUDIO_SEGMENT
    assert drafts[0].source_locator["start_ms"] == 0
    assert drafts[-1].source_locator["end_ms"] == 7500
    segmentation = drafts[0].file_info["segmentation"]
    assert segmentation["feature_window_seconds"] == 1.0
    assert segmentation["feature_window_overlap_seconds"] == 0
    assert segmentation["minimum_segment_seconds"] == 3.0
    assert segmentation["maximum_segment_seconds"] is None
    assert AudioSegmentationConfig().minimum_distance_threshold == 0.002
    assert AudioSegmentationConfig().maximum_distance_threshold == 0.02
