"""Logical audio segmentation using transient acoustic embeddings."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from capsule.enums import AssetType
from capsule.parsers.temporal_clustering import cluster_contiguous_windows
from capsule.parsers.video import VideoCancellationToken, VideoToolingError
from capsule.schemas import AssetDraft, DiscoveredFile


class AudioEmbedder(Protocol):
    """Encode independent mono waveform windows into normalized vectors."""

    def embed(self, waveforms: list[np.ndarray], *, sample_rate: int) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class AudioSegmentationConfig:
    window_seconds: float = 1.0
    minimum_segment_seconds: float = 3.0
    sample_rate: int = 32_000
    distance_quantile: float = 0.75
    minimum_distance_threshold: float = 0.002
    maximum_distance_threshold: float = 0.02
    embedding_batch_size: int = 16

    def __post_init__(self) -> None:
        if self.window_seconds != 1.0:
            raise ValueError("audio feature windows must be exactly one second")
        if self.minimum_segment_seconds <= 0 or self.sample_rate <= 0:
            raise ValueError("audio duration and sample rate parameters must be positive")
        if not 0 <= self.distance_quantile <= 1:
            raise ValueError("audio distance quantile must be between zero and one")
        if not (0 <= self.minimum_distance_threshold <= self.maximum_distance_threshold <= 2):
            raise ValueError("audio distance threshold bounds are invalid")
        if self.embedding_batch_size < 1:
            raise ValueError("audio embedding batch size must be positive")


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    duration_ms: int
    sample_rate: int | None
    channels: int | None
    codec_name: str | None
    file_size_bytes: int


class AudioParser:
    """Create logical ``audio_segment`` drafts without storing feature vectors."""

    def __init__(
        self,
        *,
        embedder: AudioEmbedder,
        config: AudioSegmentationConfig | None = None,
        concurrency: int = 1,
    ) -> None:
        if concurrency < 1:
            raise ValueError("audio parser concurrency must be positive")
        self._embedder = embedder
        self._config = config or AudioSegmentationConfig()
        self._semaphore = asyncio.Semaphore(concurrency)

    @staticmethod
    def check_dependencies() -> list[str]:
        return [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]

    async def assetize(
        self,
        source_file: DiscoveredFile,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_token: VideoCancellationToken | None = None,
    ) -> list[AssetDraft]:
        async with self._semaphore:
            return await asyncio.to_thread(
                self.assetize_path,
                Path(source_file.path),
                progress_callback=progress_callback,
                cancellation_token=cancellation_token,
            )

    def assetize_path(
        self,
        source: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        cancellation_token: VideoCancellationToken | None = None,
    ) -> list[AssetDraft]:
        token = cancellation_token or VideoCancellationToken()
        metadata = _probe_audio(source, token)
        embeddings, centers = _decode_and_embed(
            source,
            metadata=metadata,
            embedder=self._embedder,
            config=self._config,
            progress_callback=progress_callback,
            cancellation_token=token,
        )
        regions, threshold = cluster_contiguous_windows(
            embeddings,
            sample_timestamps_ms=centers,
            duration_ms=metadata.duration_ms,
            minimum_segment_seconds=self._config.minimum_segment_seconds,
            distance_quantile=self._config.distance_quantile,
            minimum_distance_threshold=self._config.minimum_distance_threshold,
            maximum_distance_threshold=self._config.maximum_distance_threshold,
        )
        return [
            AssetDraft(
                asset_type=AssetType.AUDIO_SEGMENT,
                file_name=source.name,
                source_locator={
                    "type": "time_range",
                    "segment_index": index,
                    "start_ms": region.start_ms,
                    "end_ms": region.end_ms,
                    "source": "continuous_acoustic_clustering",
                },
                file_info={
                    "audio_output_mode": "logical",
                    "duration_ms": region.end_ms - region.start_ms,
                    "source_duration_ms": metadata.duration_ms,
                    "sample_rate": metadata.sample_rate,
                    "channels": metadata.channels,
                    "codec_name": metadata.codec_name,
                    "file_size_bytes": metadata.file_size_bytes,
                    "segmentation": {
                        "feature_window_seconds": self._config.window_seconds,
                        "feature_window_overlap_seconds": 0,
                        "minimum_segment_seconds": self._config.minimum_segment_seconds,
                        "maximum_segment_seconds": None,
                        "distance_threshold": round(threshold, 6),
                    },
                },
            )
            for index, region in enumerate(regions)
        ]


def _probe_audio(source: Path, token: VideoCancellationToken) -> AudioMetadata:
    tool = shutil.which("ffprobe")
    if tool is None:
        raise VideoToolingError("ffprobe is required for audio segmentation")
    token.raise_if_cancelled()
    process = subprocess.Popen(
        [
            tool,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,codec_name,duration:format=duration,size",
            "-of",
            "json",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    unregister = token.register(process)
    try:
        stdout, stderr = process.communicate()
    finally:
        unregister()
    token.raise_if_cancelled()
    if process.returncode:
        raise VideoToolingError(stderr.decode(errors="replace").strip() or "ffprobe failed")
    payload = json.loads(stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise VideoToolingError(f"no audio stream found: {source}")
    stream = streams[0]
    format_info = payload.get("format") or {}
    duration = stream.get("duration") or format_info.get("duration")
    duration_ms = round(float(duration or 0) * 1000)
    if duration_ms <= 0:
        raise VideoToolingError(f"audio duration is unavailable: {source}")
    return AudioMetadata(
        duration_ms=duration_ms,
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=int(stream["channels"]) if stream.get("channels") else None,
        codec_name=stream.get("codec_name"),
        file_size_bytes=int(format_info.get("size") or source.stat().st_size),
    )


def _decode_and_embed(
    source: Path,
    *,
    metadata: AudioMetadata,
    embedder: AudioEmbedder,
    config: AudioSegmentationConfig,
    progress_callback: Callable[[int, int], None] | None,
    cancellation_token: VideoCancellationToken,
) -> tuple[np.ndarray, list[int]]:
    tool = shutil.which("ffmpeg")
    if tool is None:
        raise VideoToolingError("ffmpeg is required for audio segmentation")
    process = subprocess.Popen(
        [
            tool,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(config.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    unregister = cancellation_token.register(process)
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        raise VideoToolingError("could not open FFmpeg audio pipes")
    samples_per_window = round(config.window_seconds * config.sample_rate)
    bytes_per_window = samples_per_window * np.dtype("<f4").itemsize
    batches: list[np.ndarray] = []
    centers: list[int] = []
    pending: list[np.ndarray] = []
    decoded_samples = 0

    def flush() -> None:
        if not pending:
            return
        values = np.asarray(
            embedder.embed(pending.copy(), sample_rate=config.sample_rate),
            dtype=np.float32,
        )
        if values.ndim != 2 or len(values) != len(pending):
            raise VideoToolingError("audio embedder must return one vector per window")
        batches.append(values)
        pending.clear()

    try:
        while True:
            cancellation_token.raise_if_cancelled()
            chunk = _read_up_to(process.stdout, bytes_per_window)
            if not chunk:
                break
            actual_samples = len(chunk) // 4
            if actual_samples == 0:
                break
            waveform = np.frombuffer(chunk[: actual_samples * 4], dtype="<f4").copy()
            start_sample = decoded_samples
            decoded_samples += actual_samples
            centers.append(
                min(
                    metadata.duration_ms,
                    round((start_sample + actual_samples / 2) * 1000 / config.sample_rate),
                )
            )
            if actual_samples < samples_per_window:
                waveform = np.pad(waveform, (0, samples_per_window - actual_samples))
            pending.append(waveform.astype(np.float32, copy=False))
            if len(pending) >= config.embedding_batch_size:
                flush()
            if progress_callback is not None:
                progress_callback(
                    min(metadata.duration_ms, round(decoded_samples * 1000 / config.sample_rate)),
                    metadata.duration_ms,
                )
        flush()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code:
            raise VideoToolingError(
                stderr.decode(errors="replace").strip() or "FFmpeg could not decode audio"
            )
    finally:
        unregister()
        if process.poll() is None:
            cancellation_token.cancel()
        process.stdout.close()
        process.stderr.close()
    if not batches:
        raise VideoToolingError(f"audio has no decodable samples: {source}")
    return np.concatenate(batches, axis=0), centers


def _read_up_to(stream: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        value = stream.read(remaining)  # type: ignore[attr-defined]
        if not value:
            break
        chunks.append(value)
        remaining -= len(value)
    return b"".join(chunks)
