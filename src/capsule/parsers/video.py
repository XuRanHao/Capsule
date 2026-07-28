"""Video visual segmentation and Asset draft generation.

The parser owns deterministic video analysis only.  It keeps segments as time
ranges on the source file and never creates derived video files.  Visual
embeddings are supplied by a host-side backend so that the Docker application
does not need to carry PyTorch or access macOS MPS.
"""

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from capsule.enums import AssetType
from capsule.schemas import AssetDraft, DiscoveredFile


class VideoToolingError(RuntimeError):
    """Raised when FFmpeg, decoding, or the visual embedding backend fails."""


class VisualEmbedder(Protocol):
    """A batch image encoder used to select representative frames."""

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        """Return one L2-normalized embedding row per BGR OpenCV frame."""


@dataclass(frozen=True, slots=True)
class VideoSegmentationConfig:
    """First-pass parameters, all measured in source-video time."""

    scene_threshold: float = 27.0
    min_scene_seconds: float = 1.0
    max_shot_seconds: float = 45.0
    window_seconds: float = 20.0
    sample_interval_seconds: float = 5.0
    max_candidate_frames: int = 12
    max_representative_frames: int = 3
    dark_luma_max: float = 12.0
    bright_luma_min: float = 243.0
    min_luma_stddev: float = 8.0
    min_laplacian_variance: float = 20.0
    duplicate_mean_abs_diff: float = 0.015
    cluster_silhouette_min: float = 0.15

    def __post_init__(self) -> None:
        if self.min_scene_seconds <= 0 or self.max_shot_seconds <= 0:
            raise ValueError("scene durations must be positive")
        if self.window_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise ValueError("window and sample interval must be positive")
        if self.max_candidate_frames < 2 or self.max_representative_frames < 1:
            raise ValueError("at least two candidates and one representative are required")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    duration_ms: int
    fps: float
    codec_name: str | None
    pixel_format: str | None
    file_size_bytes: int


@dataclass(frozen=True, slots=True)
class VideoRange:
    scene_index: int
    window_index: int
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class CandidateFrame:
    requested_ms: int
    timestamp_ms: int
    frame: np.ndarray = field(repr=False)
    brightness: float
    contrast: float
    sharpness: float
    is_valid: bool = True
    invalid_reason: str | None = None


class VideoParser:
    """Convert a local video into logical ``video_segment`` Asset drafts."""

    def __init__(
        self,
        *,
        concurrency: int = 1,
        config: VideoSegmentationConfig | None = None,
        embedder: VisualEmbedder | None = None,
        mobileclip_model_path: Path | None = None,
        mobileclip_batch_size: int = 12,
    ) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._config = config or VideoSegmentationConfig()
        self._embedder = embedder
        self._mobileclip_model_path = mobileclip_model_path
        self._mobileclip_batch_size = mobileclip_batch_size

    @staticmethod
    def check_dependencies() -> list[str]:
        return [
            executable
            for executable in ("ffmpeg", "ffprobe")
            if shutil.which(executable) is None
        ]

    async def assetize(self, source_file: DiscoveredFile) -> list[AssetDraft]:
        """Build video Asset drafts without producing physical clips."""
        async with self._semaphore:
            return await asyncio.to_thread(self.assetize_path, Path(source_file.path))

    def assetize_path(self, source: Path) -> list[AssetDraft]:
        source = source.expanduser().resolve()
        _require_video_tools()
        metadata = _probe_metadata(source)
        ranges = _build_video_ranges(source, metadata, self._config)
        requested_times = sorted(
            {
                timestamp
                for item in ranges
                for timestamp in candidate_timestamps(
                    item.start_ms,
                    item.end_ms,
                    self._config.sample_interval_seconds,
                    self._config.max_candidate_frames,
                )
            }
        )
        candidates_by_request = _extract_candidate_frames(source, requested_times)
        embedder = self._get_embedder()

        drafts: list[AssetDraft] = []
        for segment_index, item in enumerate(ranges):
            requested = candidate_timestamps(
                item.start_ms,
                item.end_ms,
                self._config.sample_interval_seconds,
                self._config.max_candidate_frames,
            )
            candidates = [candidates_by_request[timestamp] for timestamp in requested]
            valid = filter_invalid_frames(candidates, self._config)
            selected_pool = valid or [_best_fallback(candidates)]
            embeddings = embedder.embed([candidate.frame for candidate in selected_pool])
            representatives = select_representative_frames(
                selected_pool,
                embeddings,
                self._config,
            )
            drafts.append(
                AssetDraft(
                    asset_type=AssetType.VIDEO_SEGMENT,
                    file_name=source.name,
                    source_locator={
                        "type": "time_range",
                        "segment_index": segment_index,
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                        "source": "scene_then_window",
                        "scene_indices": [item.scene_index],
                        "window_indices": [item.window_index],
                    },
                    file_info={
                        "width": metadata.width,
                        "height": metadata.height,
                        "aspect_ratio": _aspect_ratio(metadata.width, metadata.height),
                        "duration_ms": item.end_ms - item.start_ms,
                        "source_duration_ms": metadata.duration_ms,
                        "fps": metadata.fps,
                        "codec_name": metadata.codec_name,
                        "pixel_format": metadata.pixel_format,
                        "file_size_bytes": metadata.file_size_bytes,
                        "candidate_frame_count": len(candidates),
                        "valid_frame_count": len(valid),
                        "representative_frames": [
                            _frame_summary(frame) for frame in representatives
                        ],
                    },
                    raw_content=None,
                )
            )
        return drafts

    def _get_embedder(self) -> VisualEmbedder:
        if self._embedder is not None:
            return self._embedder
        from capsule.model_clients.mobileclip import MobileClipMpsEmbedder

        self._embedder = MobileClipMpsEmbedder(
            model_path=self._mobileclip_model_path,
            batch_size=self._mobileclip_batch_size,
        )
        return self._embedder


def candidate_timestamps(
    start_ms: int,
    end_ms: int,
    interval_seconds: float,
    maximum: int,
) -> list[int]:
    """Uniformly sample a range while always preserving its beginning and end."""
    if end_ms < start_ms:
        raise ValueError("end_ms must be greater than or equal to start_ms")
    if end_ms == start_ms:
        return [start_ms]

    interval_ms = max(1, round(interval_seconds * 1000))
    timestamps = list(range(start_ms, end_ms, interval_ms))
    if not timestamps or timestamps[0] != start_ms:
        timestamps.insert(0, start_ms)
    if timestamps[-1] != end_ms:
        timestamps.append(end_ms)
    if len(timestamps) <= maximum:
        return timestamps

    selected = {
        round(index * (len(timestamps) - 1) / (maximum - 1)) for index in range(maximum)
    }
    return [timestamp for index, timestamp in enumerate(timestamps) if index in selected]


def filter_invalid_frames(
    candidates: list[CandidateFrame],
    config: VideoSegmentationConfig,
) -> list[CandidateFrame]:
    """Reject obvious blank, blurry, and near-duplicate frames in time order."""
    accepted: list[CandidateFrame] = []
    previous_thumbnail: np.ndarray | None = None
    for candidate in candidates:
        reason = _invalid_reason(candidate, config)
        thumbnail = _thumbnail(candidate.frame)
        if reason is None and previous_thumbnail is not None:
            difference = float(
                np.mean(
                    np.abs(thumbnail.astype(np.float32) - previous_thumbnail.astype(np.float32))
                )
                / 255.0
            )
            if difference <= config.duplicate_mean_abs_diff:
                reason = "near_duplicate"
        candidate.is_valid = reason is None
        candidate.invalid_reason = reason
        if candidate.is_valid:
            accepted.append(candidate)
            previous_thumbnail = thumbnail
    return accepted


def select_representative_frames(
    candidates: list[CandidateFrame],
    embeddings: np.ndarray,
    config: VideoSegmentationConfig,
) -> list[CandidateFrame]:
    """Choose up to three real frames nearest to dynamic visual cluster centers."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(candidates):
        raise VideoToolingError("visual embedder must return one vector per candidate frame")
    if len(candidates) == 1:
        return candidates

    matrix = _l2_normalize(matrix)
    cluster_count = _choose_cluster_count(matrix, config)
    if cluster_count == 1:
        center = matrix.mean(axis=0, keepdims=True)
        position = int(np.argmin(np.linalg.norm(matrix - center, axis=1)))
        return [candidates[position]]

    model = KMeans(n_clusters=cluster_count, n_init="auto", random_state=0)
    labels = model.fit_predict(matrix)
    selected: list[CandidateFrame] = []
    for label in range(cluster_count):
        positions = np.flatnonzero(labels == label)
        center = model.cluster_centers_[label]
        position = positions[int(np.argmin(np.linalg.norm(matrix[positions] - center, axis=1)))]
        selected.append(candidates[int(position)])
    return sorted(selected, key=lambda candidate: candidate.timestamp_ms)


def _choose_cluster_count(matrix: np.ndarray, config: VideoSegmentationConfig) -> int:
    maximum = min(config.max_representative_frames, len(matrix) - 1)
    if maximum < 2 or np.unique(np.round(matrix, decimals=6), axis=0).shape[0] < 2:
        return 1

    best_count = 1
    best_score = config.cluster_silhouette_min
    for count in range(2, maximum + 1):
        model = KMeans(n_clusters=count, n_init="auto", random_state=0)
        labels = model.fit_predict(matrix)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(matrix, labels))
        if score > best_score:
            best_count = count
            best_score = score
    return best_count


def _build_video_ranges(
    source: Path,
    metadata: VideoMetadata,
    config: VideoSegmentationConfig,
) -> list[VideoRange]:
    scene_ranges = _detect_scenes(source, metadata, config)
    ranges: list[VideoRange] = []
    for scene_index, (start_ms, end_ms) in enumerate(scene_ranges):
        ranges.extend(
            VideoRange(scene_index, window_index, window_start, window_end)
            for window_index, (window_start, window_end) in enumerate(
                split_shot_windows(start_ms, end_ms, config)
            )
        )
    return ranges or [VideoRange(0, 0, 0, metadata.duration_ms)]


def split_shot_windows(
    start_ms: int,
    end_ms: int,
    config: VideoSegmentationConfig,
) -> list[tuple[int, int]]:
    """Keep short natural shots intact and use 20-second windows for long shots."""
    if end_ms <= start_ms:
        return []
    if end_ms - start_ms <= round(config.max_shot_seconds * 1000):
        return [(start_ms, end_ms)]
    window_ms = round(config.window_seconds * 1000)
    return [
        (window_start, min(window_start + window_ms, end_ms))
        for window_start in range(start_ms, end_ms, window_ms)
    ]


def _detect_scenes(
    source: Path,
    metadata: VideoMetadata,
    config: VideoSegmentationConfig,
) -> list[tuple[int, int]]:
    from scenedetect import ContentDetector, detect

    minimum_frames = max(1, round(metadata.fps * config.min_scene_seconds))
    detected = detect(
        str(source),
        ContentDetector(threshold=config.scene_threshold, min_scene_len=minimum_frames),
        show_progress=False,
    )
    ranges = [
        (
            max(0, round(start.get_seconds() * 1000)),
            min(metadata.duration_ms, round(end.get_seconds() * 1000)),
        )
        for start, end in detected
        if end.get_seconds() > start.get_seconds()
    ]
    return ranges or [(0, metadata.duration_ms)]


def _probe_metadata(source: Path) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,codec_name,pix_fmt,duration",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    if result.returncode:
        raise VideoToolingError(result.stderr.strip() or f"ffprobe failed for {source}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise VideoToolingError(f"no video stream found: {source}")
    stream = streams[0]
    format_info = payload.get("format", {})
    duration = _float_or_none(stream.get("duration")) or _float_or_none(format_info.get("duration"))
    if duration is None or duration <= 0:
        raise VideoToolingError(f"video duration is unavailable: {source}")
    return VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_ms=max(1, round(duration * 1000)),
        fps=_parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        codec_name=_string_or_none(stream.get("codec_name")),
        pixel_format=_string_or_none(stream.get("pix_fmt")),
        file_size_bytes=int(format_info.get("size", source.stat().st_size)),
    )


def _extract_candidate_frames(
    source: Path,
    requested_times: list[int],
) -> dict[int, CandidateFrame]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolingError(f"OpenCV cannot decode video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise VideoToolingError(f"video FPS is unavailable: {source}")

    candidates: dict[int, CandidateFrame] = {}
    next_index = 0
    frame_index = 0
    previous: tuple[int, np.ndarray] | None = None
    try:
        while next_index < len(requested_times):
            succeeded, frame = capture.read()
            if not succeeded:
                break
            timestamp_ms = round(frame_index * 1000 / fps)
            current = (timestamp_ms, frame)
            while next_index < len(requested_times) and timestamp_ms >= requested_times[next_index]:
                target = requested_times[next_index]
                selected_time, selected_frame = _nearest_frame(
                    previous,
                    (timestamp_ms, frame),
                    target,
                )
                candidates[target] = _candidate_frame(target, selected_time, selected_frame)
                next_index += 1
            previous = current
            frame_index += 1

        if previous is None:
            raise VideoToolingError(f"video has no decodable frames: {source}")
        while next_index < len(requested_times):
            target = requested_times[next_index]
            candidates[target] = _candidate_frame(target, previous[0], previous[1])
            next_index += 1
    finally:
        capture.release()
    return candidates


def _candidate_frame(requested_ms: int, timestamp_ms: int, frame: np.ndarray) -> CandidateFrame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return CandidateFrame(
        requested_ms=requested_ms,
        timestamp_ms=timestamp_ms,
        frame=frame.copy(),
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )


def _nearest_frame(
    previous: tuple[int, np.ndarray] | None,
    current: tuple[int, np.ndarray],
    target: int,
) -> tuple[int, np.ndarray]:
    if previous is None or abs(current[0] - target) <= abs(previous[0] - target):
        return current
    return previous


def _invalid_reason(candidate: CandidateFrame, config: VideoSegmentationConfig) -> str | None:
    if candidate.brightness <= config.dark_luma_max:
        return "black_frame"
    if candidate.brightness >= config.bright_luma_min:
        return "overexposed_frame"
    if candidate.contrast < config.min_luma_stddev:
        return "low_contrast_frame"
    if candidate.sharpness < config.min_laplacian_variance:
        return "blurry_frame"
    return None


def _best_fallback(candidates: list[CandidateFrame]) -> CandidateFrame:
    if not candidates:
        raise VideoToolingError("no candidate frame was extracted")
    return max(candidates, key=lambda candidate: candidate.sharpness)


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    return np.asarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)))


def _frame_summary(candidate: CandidateFrame) -> dict[str, float | int | str | None]:
    return {
        "timestamp_ms": candidate.timestamp_ms,
        "brightness": round(candidate.brightness, 3),
        "contrast": round(candidate.contrast, 3),
        "sharpness": round(candidate.sharpness, 3),
        "invalid_reason": candidate.invalid_reason,
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return cast(np.ndarray, matrix / np.maximum(norms, np.finfo(np.float32).eps))


def _parse_frame_rate(value: object) -> float:
    text = str(value)
    numerator, separator, denominator = text.partition("/")
    if separator:
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    return float(text)


def _float_or_none(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _aspect_ratio(width: int, height: int) -> float | None:
    return round(width / height, 6) if height else None


def _require_video_tools() -> None:
    missing = VideoParser.check_dependencies()
    if missing:
        raise VideoToolingError(f"missing video dependencies: {', '.join(missing)}")
