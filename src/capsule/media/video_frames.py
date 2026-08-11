"""Ephemeral representative-frame extraction for logical video Assets."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from capsule.parsers.video import resolve_video_tool


class VideoFrameExtractionError(ValueError):
    """Representative frames cannot be read without persisting media."""


@dataclass(frozen=True, slots=True)
class VideoFrameRequest:
    source_uri: str
    timestamps_ms: tuple[int, ...]
    max_edge: int = 768


class VideoFrameExtractor(Protocol):
    async def extract(self, request: VideoFrameRequest) -> list[bytes]: ...


def logical_video_frame_request(
    *,
    source_uri: str,
    file_info: Mapping[str, Any],
    source_locator: Mapping[str, Any],
    max_frames: int = 3,
) -> VideoFrameRequest:
    """Build a bounded extraction request from durable timestamp metadata."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    start_ms = _integer(source_locator.get("start_ms"), default=0)
    end_ms = _integer(source_locator.get("end_ms"), default=start_ms + 1)
    representatives = file_info.get("representative_frames")
    timestamps: list[int] = []
    if isinstance(representatives, list):
        for item in representatives:
            if not isinstance(item, Mapping):
                continue
            timestamp_ms = _integer(item.get("timestamp_ms"), default=-1)
            if timestamp_ms < start_ms or timestamp_ms >= end_ms:
                continue
            if timestamp_ms not in timestamps:
                timestamps.append(timestamp_ms)
            if len(timestamps) >= max_frames:
                break
    if not timestamps:
        timestamps.append(start_ms + max(0, end_ms - start_ms - 1) // 2)
    return VideoFrameRequest(
        source_uri=source_uri,
        timestamps_ms=tuple(timestamps),
    )


class FFmpegVideoFrameExtractor:
    """Read representative JPEGs through stdout and never create a media file."""

    def __init__(self, *, concurrency: int = 2, timeout_seconds: float = 30.0) -> None:
        if concurrency < 1 or timeout_seconds <= 0:
            raise ValueError("video frame extraction limits must be positive")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout_seconds = timeout_seconds
        self._inflight_lock = asyncio.Lock()
        self._inflight: dict[VideoFrameRequest, asyncio.Task[list[bytes]]] = {}

    async def extract(self, request: VideoFrameRequest) -> list[bytes]:
        # Understanding and native embedding start together. Share only their
        # concurrent decode, then evict immediately; this is not a frame cache.
        async with self._inflight_lock:
            task = self._inflight.get(request)
            if task is None:
                task = asyncio.create_task(self._extract_and_evict(request))
                task.add_done_callback(_consume_task_exception)
                self._inflight[request] = task
        return await asyncio.shield(task)

    async def _extract_and_evict(self, request: VideoFrameRequest) -> list[bytes]:
        try:
            return await self._extract_uncached(request)
        finally:
            current = asyncio.current_task()
            async with self._inflight_lock:
                if self._inflight.get(request) is current:
                    self._inflight.pop(request, None)

    async def _extract_uncached(self, request: VideoFrameRequest) -> list[bytes]:
        parsed = urlparse(request.source_uri)
        if parsed.scheme != "file":
            raise VideoFrameExtractionError(
                "logical video frame extraction currently requires a local source"
            )
        source = await asyncio.to_thread(_resolve_existing_source, unquote(parsed.path))
        ffmpeg = resolve_video_tool("ffmpeg")
        if ffmpeg is None:
            raise VideoFrameExtractionError("FFmpeg is unavailable for logical video frames")
        async with self._semaphore:
            return [
                await self._extract_one(
                    ffmpeg=ffmpeg,
                    source=source,
                    timestamp_ms=timestamp_ms,
                    max_edge=request.max_edge,
                )
                for timestamp_ms in request.timestamps_ms
            ]

    async def _extract_one(
        self,
        *,
        ffmpeg: Path,
        source: Path,
        timestamp_ms: int,
        max_edge: int,
    ) -> bytes:
        process = await asyncio.create_subprocess_exec(
            str(ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{timestamp_ms / 1_000:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease",
            "-c:v",
            "mjpeg",
            "-q:v",
            "3",
            "-f",
            "image2pipe",
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_seconds,
            )
        except BaseException:
            await _terminate_process_group(process)
            raise
        if process.returncode:
            detail = stderr.decode(errors="replace").strip() or "FFmpeg frame extraction failed"
            raise VideoFrameExtractionError(detail[:1_000])
        if not stdout.startswith(b"\xff\xd8"):
            raise VideoFrameExtractionError("FFmpeg returned an invalid representative JPEG")
        return stdout


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def _integer(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return round(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _consume_task_exception(task: asyncio.Task[list[bytes]]) -> None:
    if not task.cancelled():
        task.exception()


def _resolve_existing_source(raw_path: str) -> Path:
    source = Path(raw_path).resolve()
    if not source.is_file():
        raise VideoFrameExtractionError(f"video source file no longer exists: {source}")
    return source
