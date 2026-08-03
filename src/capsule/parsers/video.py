"""Video visual segmentation and Asset draft generation.

The parser owns deterministic video analysis and creates logical Segment
drafts. The pipeline's media writer turns those drafts into persistent clips
and images. Visual embeddings are supplied by a host-side backend so that the
Docker application does not need to carry PyTorch or access macOS MPS.
"""

import asyncio
import json
import math
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol, cast

import cv2
import numpy as np
from scipy.sparse import diags  # type: ignore[import-untyped]
from sklearn.cluster import AgglomerativeClustering, KMeans
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
    """Adaptive content segmentation and representative-frame parameters."""

    sample_interval_seconds: float = 0.5
    min_segment_seconds: float = 1.0
    distance_quantile: float = 0.75
    min_distance_threshold: float = 0.08
    max_distance_threshold: float = 0.25
    similarity_relaxation: float = 0.05
    max_merge_cost: float = 0.5
    base_target_seconds: float = 10.0
    max_target_seconds: float = 20.0
    target_log2_weight: float = 0.15
    hard_max_duration_factor: float = 1.5
    activity_sample_fps: float = 6.0
    activity_envelope_seconds: float = 0.5
    activity_shift_side_seconds: float = 1.0
    keyframe_size: int = 224
    keyframe_jpeg_quality: int = 85
    max_representative_frames: int = 3
    dark_luma_max: float = 12.0
    bright_luma_min: float = 243.0
    min_luma_stddev: float = 8.0
    min_laplacian_variance: float = 20.0
    duplicate_mean_abs_diff: float = 0.015
    cluster_silhouette_min: float = 0.15

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0 or self.min_segment_seconds <= 0:
            raise ValueError("video sampling and minimum segment durations must be positive")
        if not 0 <= self.distance_quantile <= 1:
            raise ValueError("video distance quantile must be between zero and one")
        if not 0 <= self.min_distance_threshold <= self.max_distance_threshold <= 2:
            raise ValueError("video distance threshold bounds are invalid")
        if not 0 <= self.similarity_relaxation <= 2 or self.max_merge_cost < 0:
            raise ValueError("video second-stage merge parameters are invalid")
        if min(self.base_target_seconds, self.max_target_seconds) <= 0:
            raise ValueError("video target durations must be positive")
        if self.base_target_seconds > self.max_target_seconds:
            raise ValueError("video base target duration cannot exceed its maximum")
        if self.target_log2_weight < 0 or self.hard_max_duration_factor < 1:
            raise ValueError("video adaptive duration parameters are invalid")
        if self.activity_sample_fps <= 0:
            raise ValueError("video activity sample rate must be positive")
        if self.activity_sample_fps * self.sample_interval_seconds < 1:
            raise ValueError("video activity rate must cover the content sample interval")
        if self.keyframe_size != 224:
            raise ValueError("video keyframes must be exactly 224x224")
        if not 1 <= self.keyframe_jpeg_quality <= 100:
            raise ValueError("video keyframe JPEG quality must be between 1 and 100")
        if self.max_representative_frames < 1:
            raise ValueError("at least one representative frame is required")


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
    atom_indices: tuple[int, ...]
    start_ms: int
    end_ms: int


@dataclass(slots=True)
class CandidateFrame:
    requested_ms: int
    timestamp_ms: int
    frame: np.ndarray | None = field(repr=False)
    brightness: float
    contrast: float
    sharpness: float
    jpeg_bytes: bytes | None = field(default=None, repr=False)
    thumbnail: np.ndarray | None = field(default=None, repr=False)
    is_valid: bool = True
    invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _VideoAnalysis:
    candidates: list[CandidateFrame]
    embeddings: np.ndarray
    activity_envelope: np.ndarray


@dataclass(frozen=True, slots=True)
class _Region:
    atom_indices: tuple[int, ...]
    sample_indices: tuple[int, ...]
    start_ms: int
    end_ms: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000

    def centroid(self, embeddings: np.ndarray) -> np.ndarray:
        value = embeddings[list(self.sample_indices)].mean(axis=0)
        return cast(
            np.ndarray,
            value / max(float(np.linalg.norm(value)), np.finfo(np.float32).eps),
        )


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
        if concurrency < 1 or mobileclip_batch_size < 1:
            raise ValueError("video concurrency and MobileCLIP batch size must be positive")
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
            if resolve_video_tool(executable) is None
        ]

    async def assetize(self, source_file: DiscoveredFile) -> list[AssetDraft]:
        """Build logical video Segment drafts for later media persistence."""
        async with self._semaphore:
            return await asyncio.to_thread(self.assetize_path, Path(source_file.path))

    def assetize_path(self, source: Path) -> list[AssetDraft]:
        source = source.expanduser().resolve()
        _require_video_tools()
        metadata = _probe_metadata(source)
        analysis = _analyze_video(
            source,
            metadata,
            self._config,
            self._get_embedder(),
            embedding_batch_size=self._mobileclip_batch_size,
        )
        atoms, distance_threshold = _content_atoms(analysis, metadata, self._config)
        ranges, merge_info = _merge_atoms(
            atoms,
            analysis,
            metadata,
            self._config,
            distance_threshold=distance_threshold,
        )
        candidate_positions = {
            id(candidate): index
            for index, candidate in enumerate(analysis.candidates)
        }

        drafts: list[AssetDraft] = []
        for segment_index, item in enumerate(ranges):
            positions = _candidate_positions_for_range(
                analysis.candidates,
                item.start_ms,
                item.end_ms,
                include_end=segment_index == len(ranges) - 1,
            )
            candidates = [analysis.candidates[position] for position in positions]
            valid = filter_invalid_frames(candidates, self._config)
            selected_pool = valid or [_best_fallback(candidates)]
            selected_positions = [candidate_positions[id(candidate)] for candidate in selected_pool]
            representatives = select_representative_frames(
                selected_pool,
                analysis.embeddings[selected_positions],
                self._config,
            )
            keyframe_jpegs = [
                _required_jpeg(candidate)
                for candidate in representatives
            ]
            drafts.append(
                AssetDraft(
                    asset_type=AssetType.VIDEO_SEGMENT,
                    file_name=source.name,
                    source_locator={
                        "type": "time_range",
                        "segment_index": segment_index,
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                        "source": "adaptive_content_two_stage",
                        "atom_indices": [index + 1 for index in item.atom_indices],
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
                        "segmentation": {
                            "first_stage_distance_threshold": round(distance_threshold, 6),
                            "second_stage_similarity_gate": round(
                                float(merge_info["similarity_gate"]), 6
                            ),
                            "adaptive_target_seconds": round(
                                float(merge_info["duration_target"]), 6
                            ),
                            "max_merge_cost": self._config.max_merge_cost,
                        },
                    },
                    raw_content=None,
                    transient_keyframe_jpegs=keyframe_jpegs,
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


def filter_invalid_frames(
    candidates: list[CandidateFrame],
    config: VideoSegmentationConfig,
) -> list[CandidateFrame]:
    """Reject obvious blank, blurry, and near-duplicate frames in time order."""
    accepted: list[CandidateFrame] = []
    previous_thumbnail: np.ndarray | None = None
    for candidate in candidates:
        reason = _invalid_reason(candidate, config)
        thumbnail = candidate.thumbnail
        if thumbnail is None:
            if candidate.frame is None:
                raise VideoToolingError("video candidate is missing its duplicate thumbnail")
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


def _analyze_video(
    source: Path,
    metadata: VideoMetadata,
    config: VideoSegmentationConfig,
    embedder: VisualEmbedder,
    *,
    embedding_batch_size: int,
) -> _VideoAnalysis:
    frame_size = config.keyframe_size
    activity_fps = config.activity_sample_fps
    filter_graph = (
        f"fps={activity_fps:.6f},split=2[activity_source][keyframe_source];"
        f"[activity_source]scale={frame_size}:{frame_size}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={frame_size}:{frame_size}:(ow-iw)/2:(oh-ih)/2[activity];"
        f"[keyframe_source]scale={frame_size}:{frame_size}:"
        "force_original_aspect_ratio=increase,"
        f"crop={frame_size}:{frame_size}[keyframe];"
        "[activity][keyframe]hstack=inputs=2"
    )
    command = [str(_require_video_tool("ffmpeg")), "-v", "error", "-nostdin"]
    if platform.system() == "Darwin":
        command.extend(["-hwaccel", "videotoolbox"])
    command.extend(
        [
            "-i",
            str(source),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise VideoToolingError("FFmpeg did not expose its video analysis pipes")

    candidates: list[CandidateFrame] = []
    embedding_batches: list[np.ndarray] = []
    pending_candidates: list[CandidateFrame] = []
    pending_frames: list[np.ndarray] = []
    activity_scores: list[float] = []
    previous_activity_frame: np.ndarray | None = None
    frame_index = 0
    next_sample_ms = 0
    bytes_per_frame = frame_size * frame_size * 2 * 3

    def flush_embeddings() -> None:
        if not pending_frames:
            return
        embedded = np.asarray(embedder.embed(pending_frames), dtype=np.float32)
        if embedded.ndim != 2 or embedded.shape[0] != len(pending_frames):
            raise VideoToolingError("visual embedder must return one vector per sampled frame")
        embedding_batches.append(_l2_normalize(embedded))
        for candidate in pending_candidates:
            candidate.frame = None
        pending_candidates.clear()
        pending_frames.clear()

    try:
        while True:
            payload = _read_exact(process.stdout, bytes_per_frame)
            if not payload:
                break
            if len(payload) != bytes_per_frame:
                raise VideoToolingError("FFmpeg returned a truncated video analysis frame")
            combined = np.frombuffer(payload, dtype=np.uint8).reshape(
                frame_size,
                frame_size * 2,
                3,
            )
            activity_frame = cv2.cvtColor(combined[:, :frame_size], cv2.COLOR_BGR2HSV)
            activity_scores.append(
                _continuous_content_score(previous_activity_frame, activity_frame)
            )
            previous_activity_frame = activity_frame
            timestamp_ms = min(metadata.duration_ms, round(frame_index * 1000 / activity_fps))
            if timestamp_ms >= next_sample_ms:
                keyframe = np.ascontiguousarray(combined[:, frame_size:])
                candidate = _candidate_frame(
                    timestamp_ms,
                    timestamp_ms,
                    keyframe,
                    jpeg_quality=config.keyframe_jpeg_quality,
                )
                candidates.append(candidate)
                pending_candidates.append(candidate)
                assert candidate.frame is not None
                pending_frames.append(candidate.frame)
                if len(pending_frames) >= max(1, embedding_batch_size):
                    flush_embeddings()
                next_sample_ms += max(1, round(config.sample_interval_seconds * 1000))
            frame_index += 1
        flush_embeddings()
        process.stdout.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code:
            message = stderr.decode(errors="replace").strip()
            raise VideoToolingError(message or f"FFmpeg could not decode video: {source}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    if not candidates or not embedding_batches:
        raise VideoToolingError(f"video has no decodable sample frames: {source}")
    scores = np.asarray(activity_scores, dtype=np.float64)
    radius = max(1, round(config.activity_envelope_seconds * activity_fps / 2))
    envelope = _rolling_max(scores, radius)
    return _VideoAnalysis(
        candidates=candidates,
        embeddings=np.concatenate(embedding_batches, axis=0),
        activity_envelope=envelope,
    )


def _content_atoms(
    analysis: _VideoAnalysis,
    metadata: VideoMetadata,
    config: VideoSegmentationConfig,
) -> tuple[list[_Region], float]:
    embeddings = analysis.embeddings
    if len(embeddings) == 1:
        return [_Region((0,), (0,), 0, metadata.duration_ms)], config.min_distance_threshold
    distances = 1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    threshold = float(np.clip(
        np.quantile(distances, config.distance_quantile),
        config.min_distance_threshold,
        config.max_distance_threshold,
    ))
    size = len(embeddings)
    connectivity = diags([np.ones(size - 1), np.ones(size - 1)], [-1, 1], shape=(size, size))
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
        connectivity=connectivity,
        compute_full_tree=True,
    ).fit_predict(embeddings)
    minimum_samples = max(1, round(config.min_segment_seconds / config.sample_interval_seconds))
    labels = _merge_short_runs(labels, embeddings, minimum_samples)
    starts, ends = _runs(labels)
    boundaries = [0]
    for start in starts[1:]:
        left = analysis.candidates[int(start) - 1].timestamp_ms
        right = analysis.candidates[int(start)].timestamp_ms
        boundaries.append(round((left + right) / 2))
    boundaries.append(metadata.duration_ms)
    atoms = [
        _Region(
            atom_indices=(index,),
            sample_indices=tuple(range(int(start), int(end))),
            start_ms=boundaries[index],
            end_ms=boundaries[index + 1],
        )
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]
    atoms = _merge_short_regions(atoms, embeddings, config.min_segment_seconds)
    return [
        _Region((index,), atom.sample_indices, atom.start_ms, atom.end_ms)
        for index, atom in enumerate(atoms)
    ], threshold


def _merge_atoms(
    atoms: list[_Region],
    analysis: _VideoAnalysis,
    metadata: VideoMetadata,
    config: VideoSegmentationConfig,
    *,
    distance_threshold: float,
) -> tuple[list[VideoRange], dict[str, float]]:
    duration_seconds = metadata.duration_ms / 1000
    duration_target = float(np.clip(
        config.base_target_seconds
        * (
            1.0
            + config.target_log2_weight
            * math.log2(
                max(duration_seconds, config.base_target_seconds)
                / config.base_target_seconds
            )
        ),
        config.base_target_seconds,
        config.max_target_seconds,
    ))
    similarity_gate = float(np.clip(1.0 - distance_threshold - config.similarity_relaxation, -1, 1))
    if len(atoms) == 1:
        only = atoms[0]
        return [VideoRange(only.atom_indices, only.start_ms, only.end_ms)], {
            "duration_target": duration_target,
            "similarity_gate": similarity_gate,
        }

    raw_shifts = []
    side = max(1, round(config.activity_shift_side_seconds * config.activity_sample_fps))
    envelope = analysis.activity_envelope
    for atom in atoms[:-1]:
        boundary_frame = int(
            np.clip(
                round(atom.end_ms * config.activity_sample_fps / 1000),
                0,
                len(envelope) - 1,
            )
        )
        left_window = envelope[max(0, boundary_frame - side) : boundary_frame]
        right_window = envelope[boundary_frame : min(len(envelope), boundary_frame + side)]
        left_level = float(np.median(left_window)) if len(left_window) else 0.0
        right_level = float(np.median(right_window)) if len(right_window) else 0.0
        raw_shifts.append(abs(math.log1p(right_level) - math.log1p(left_level)))
    shift_values = np.asarray(raw_shifts, dtype=np.float64)
    shift_low, shift_high = np.quantile(shift_values, [.10, .90])
    shift_costs = np.clip(
        (shift_values - shift_low) / max(float(shift_high - shift_low), np.finfo(float).eps),
        0,
        1,
    )
    hard_max = config.hard_max_duration_factor * duration_target
    regions = atoms.copy()
    while len(regions) > 1:
        eligible: list[tuple[float, int]] = []
        for index, (left_region, right_region) in enumerate(
            zip(regions[:-1], regions[1:], strict=True)
        ):
            total_duration = left_region.duration_seconds + right_region.duration_seconds
            similarity = float(
                left_region.centroid(analysis.embeddings)
                @ right_region.centroid(analysis.embeddings)
            )
            if similarity < similarity_gate or total_duration > hard_max:
                continue
            boundary_id = left_region.atom_indices[-1]
            duration_cost = (total_duration / duration_target) ** 2
            merge_cost = 0.5 * duration_cost + 0.5 * float(shift_costs[boundary_id])
            eligible.append((merge_cost, index))
        if not eligible:
            break
        best_cost, best_index = min(eligible)
        if best_cost > config.max_merge_cost:
            break
        left_region, right_region = regions[best_index], regions[best_index + 1]
        regions[best_index : best_index + 2] = [
            _Region(
                atom_indices=left_region.atom_indices + right_region.atom_indices,
                sample_indices=left_region.sample_indices + right_region.sample_indices,
                start_ms=left_region.start_ms,
                end_ms=right_region.end_ms,
            )
        ]
    return [
        VideoRange(region.atom_indices, region.start_ms, region.end_ms)
        for region in regions
    ], {"duration_target": duration_target, "similarity_gate": similarity_gate}


def _candidate_positions_for_range(
    candidates: list[CandidateFrame],
    start_ms: int,
    end_ms: int,
    *,
    include_end: bool,
) -> list[int]:
    positions = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.timestamp_ms >= start_ms
        and (candidate.timestamp_ms <= end_ms if include_end else candidate.timestamp_ms < end_ms)
    ]
    if positions:
        return positions
    midpoint = (start_ms + end_ms) / 2
    return [
        min(
            range(len(candidates)),
            key=lambda index: abs(candidates[index].timestamp_ms - midpoint),
        )
    ]


def _merge_short_regions(
    regions: list[_Region],
    embeddings: np.ndarray,
    minimum_seconds: float,
) -> list[_Region]:
    result = regions.copy()
    while len(result) > 1:
        short = next(
            (
                index
                for index, region in enumerate(result)
                if region.duration_seconds < minimum_seconds
            ),
            None,
        )
        if short is None:
            return result
        center = result[short].centroid(embeddings)
        choices: list[tuple[float, int]] = []
        if short:
            choices.append(
                (
                    1.0 - float(center @ result[short - 1].centroid(embeddings)),
                    short - 1,
                )
            )
        if short + 1 < len(result):
            choices.append(
                (
                    1.0 - float(center @ result[short + 1].centroid(embeddings)),
                    short + 1,
                )
            )
        neighbor = min(choices)[1]
        left_index, right_index = sorted((short, neighbor))
        left_region, right_region = result[left_index], result[right_index]
        result[left_index : right_index + 1] = [
            _Region(
                atom_indices=left_region.atom_indices + right_region.atom_indices,
                sample_indices=left_region.sample_indices + right_region.sample_indices,
                start_ms=left_region.start_ms,
                end_ms=right_region.end_ms,
            )
        ]
    return result


def _runs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return starts, np.r_[starts[1:], len(labels)]


def _merge_short_runs(
    labels: np.ndarray,
    embeddings: np.ndarray,
    minimum_samples: int,
) -> np.ndarray:
    result = labels.copy()
    while True:
        starts, ends = _runs(result)
        short = next(
            (
                index
                for index, (start, end) in enumerate(zip(starts, ends, strict=True))
                if end - start < minimum_samples
            ),
            None,
        )
        if short is None or len(starts) == 1:
            return result
        start, end = int(starts[short]), int(ends[short])
        center = _normalized_centroid(embeddings[start:end])
        choices: list[tuple[float, int]] = []
        if short:
            left = _normalized_centroid(embeddings[starts[short - 1] : ends[short - 1]])
            choices.append((1.0 - float(center @ left), int(result[start - 1])))
        if short + 1 < len(starts):
            right = _normalized_centroid(embeddings[starts[short + 1] : ends[short + 1]])
            choices.append((1.0 - float(center @ right), int(result[end])))
        result[start:end] = min(choices)[1]


def _normalized_centroid(values: np.ndarray) -> np.ndarray:
    value = values.mean(axis=0)
    return cast(
        np.ndarray,
        value / max(float(np.linalg.norm(value)), np.finfo(np.float32).eps),
    )


def _rolling_max(values: np.ndarray, radius: int) -> np.ndarray:
    return np.asarray([
        np.max(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        for index in range(len(values))
    ])


def _probe_metadata(source: Path) -> VideoMetadata:
    command = [
        str(_require_video_tool("ffprobe")),
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
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoToolingError(f"ffprobe returned invalid metadata for {source}") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise VideoToolingError(f"no video stream found: {source}")
    stream = streams[0]
    format_info = payload.get("format", {})
    duration = _float_or_none(stream.get("duration")) or _float_or_none(format_info.get("duration"))
    if duration is None or duration <= 0:
        raise VideoToolingError(f"video duration is unavailable: {source}")
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        file_size_bytes = int(format_info.get("size", source.stat().st_size))
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoToolingError(f"video dimensions or file size are unavailable: {source}") from exc
    if width <= 0 or height <= 0 or file_size_bytes < 0:
        raise VideoToolingError(f"video dimensions or file size are invalid: {source}")
    return VideoMetadata(
        width=width,
        height=height,
        duration_ms=max(1, round(duration * 1000)),
        fps=_parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        codec_name=_string_or_none(stream.get("codec_name")),
        pixel_format=_string_or_none(stream.get("pix_fmt")),
        file_size_bytes=file_size_bytes,
    )


def _candidate_frame(
    requested_ms: int,
    timestamp_ms: int,
    frame: np.ndarray,
    *,
    jpeg_quality: int = 85,
) -> CandidateFrame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    encoded, payload = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not encoded:
        raise VideoToolingError("OpenCV could not encode a sampled video keyframe")
    return CandidateFrame(
        requested_ms=requested_ms,
        timestamp_ms=timestamp_ms,
        frame=np.ascontiguousarray(frame),
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        jpeg_bytes=payload.tobytes(),
        thumbnail=np.asarray(cv2.resize(gray, (32, 32))),
    )


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _continuous_content_score(
    previous_hsv: np.ndarray | None,
    current_hsv: np.ndarray,
) -> float:
    """Return change between two HSV frames without applying scene-cut logic.

    This stable OpenCV implementation uses mean absolute H/S/V channel
    differences with equal weights.  It deliberately has no threshold,
    minimum scene length, or cut filtering.
    """
    if previous_hsv is None:
        return 0.0
    if previous_hsv.shape != current_hsv.shape:
        current_hsv = cv2.resize(
            current_hsv,
            (previous_hsv.shape[1], previous_hsv.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    channel_difference = cv2.absdiff(previous_hsv, current_hsv)
    return float(np.mean(channel_difference, dtype=np.float64))


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


def _required_jpeg(candidate: CandidateFrame) -> bytes:
    if candidate.jpeg_bytes is None:
        raise VideoToolingError("selected video representative is missing its cached JPEG")
    return candidate.jpeg_bytes


def _frame_summary(candidate: CandidateFrame) -> dict[str, float | int | str | None]:
    return {
        "timestamp_ms": candidate.timestamp_ms,
        "brightness": round(candidate.brightness, 3),
        "contrast": round(candidate.contrast, 3),
        "sharpness": round(candidate.sharpness, 3),
        "invalid_reason": candidate.invalid_reason,
    }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise VideoToolingError("visual embedder returned non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise VideoToolingError("visual embedder returned a zero-length vector")
    return cast(np.ndarray, matrix / norms)


def _parse_frame_rate(value: object) -> float:
    try:
        text = str(value)
        numerator, separator, denominator = text.partition("/")
        fps = (
            float(numerator) / float(denominator)
            if separator and float(denominator)
            else float(text)
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise VideoToolingError("video FPS is unavailable") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise VideoToolingError("video FPS is unavailable")
    return fps


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


def _require_video_tool(executable: str) -> Path:
    resolved = resolve_video_tool(executable)
    if resolved is None:
        raise VideoToolingError(f"missing video dependency: {executable}")
    return resolved


def resolve_video_tool(executable: str) -> Path | None:
    if found := shutil.which(executable):
        return Path(found)
    if platform.system() != "Darwin":
        return None
    project_tool = Path.cwd() / "tmp" / "tools" / "ffmpeg" / "bin" / executable
    return project_tool if project_tool.is_file() and project_tool.stat().st_mode & 0o111 else None
