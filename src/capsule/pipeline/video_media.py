"""Persist the playable artifacts belonging to analyzed video Segment Assets."""

import asyncio
import subprocess
import tempfile
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

from capsule.enums import AssetType
from capsule.parsers.video import VideoToolingError, resolve_video_tool
from capsule.schemas import AssetCreate, DiscoveredFile


class VideoArtifactStorage(Protocol):
    """The object-storage operations needed to persist video-derived media."""

    async def ensure_bucket(self) -> None: ...

    async def upload_file(
        self,
        source: Path,
        object_key: str,
        *,
        content_type: str | None = None,
    ) -> str: ...


class VideoDerivedMediaWriter:
    """Create a playable clip, cover image, and keyframe images for video Assets.

    The database Asset stays a logical time range. These files make that range
    playable and can later be supplied unchanged to video embedding calls.
    """

    def __init__(self, storage: VideoArtifactStorage, *, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._storage = storage
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._bucket_lock = asyncio.Lock()
        self._bucket_ready = False

    async def persist(
        self,
        *,
        source_file: DiscoveredFile,
        assets: Sequence[AssetCreate],
    ) -> list[AssetCreate]:
        video_assets = [asset for asset in assets if asset.asset_type is AssetType.VIDEO_SEGMENT]
        if not video_assets:
            return list(assets)

        source = await asyncio.to_thread(_resolve_source, source_file)
        await self._ensure_bucket()

        with tempfile.TemporaryDirectory(prefix="capsule-video-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            persisted = await self._persist_many(
                source=source,
                assets=video_assets,
                temporary_root=temporary_root,
            )
        updated = {asset.asset_key: asset for asset in persisted}
        return [updated.get(asset.asset_key, asset) for asset in assets]

    async def _persist_many(
        self,
        *,
        source: Path,
        assets: list[AssetCreate],
        temporary_root: Path,
    ) -> list[AssetCreate]:
        persisted: list[AssetCreate | None] = [None] * len(assets)
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(assets):
                index = next_index
                next_index += 1
                asset = assets[index]
                persisted[index] = await self._persist_one(
                    source=source,
                    asset=asset,
                    output_directory=temporary_root / asset.asset_key,
                )

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._concurrency, len(assets)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        if any(asset is None for asset in persisted):
            raise RuntimeError("video media persistence ended before all assets completed")
        return [asset for asset in persisted if asset is not None]

    async def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        async with self._bucket_lock:
            if not self._bucket_ready:
                await self._storage.ensure_bucket()
                self._bucket_ready = True

    async def _persist_one(
        self,
        *,
        source: Path,
        asset: AssetCreate,
        output_directory: Path,
    ) -> AssetCreate:
        async with self._semaphore:
            artifacts = await asyncio.to_thread(
                _render_artifacts,
                source,
                asset,
                output_directory,
            )
            prefix = _object_prefix(asset)
            uploads = await asyncio.gather(
                self._storage.upload_file(
                    artifacts.segment_path,
                    f"{prefix}/segment.mp4",
                    content_type="video/mp4",
                ),
                self._storage.upload_file(
                    artifacts.keyframe_paths[0],
                    f"{prefix}/preview.jpg",
                    content_type="image/jpeg",
                ),
                *(
                    self._storage.upload_file(
                        frame_path,
                        f"{prefix}/keyframes/{ordinal:02d}.jpg",
                        content_type="image/jpeg",
                    )
                    for ordinal, frame_path in enumerate(
                        artifacts.keyframe_paths,
                        start=1,
                    )
                ),
            )
            derived_uri, preview_uri, *frame_uris = uploads
            original_summaries = _representative_summaries(asset)
            keyframes: list[dict[str, object]] = []
            for ordinal, frame_uri in enumerate(frame_uris, start=1):
                summary = original_summaries[ordinal - 1]
                keyframes.append(
                    {
                        "uri": frame_uri,
                        "timestamp_ms": summary["timestamp_ms"],
                        "role": "representative",
                        "brightness": summary.get("brightness"),
                        "contrast": summary.get("contrast"),
                        "sharpness": summary.get("sharpness"),
                    }
                )
            file_info = dict(asset.file_info)
            file_info["keyframes"] = keyframes
            file_info["derived_video"] = {
                "container": "mp4",
                "video_codec": artifacts.video_encoder,
                "audio_codec": "aac",
            }
            return asset.model_copy(
                update={
                    "derived_file_uri": derived_uri,
                    "preview_uri": preview_uri,
                    "file_info": file_info,
                }
            )


class _RenderedArtifacts:
    def __init__(
        self,
        *,
        segment_path: Path,
        keyframe_paths: list[Path],
        video_encoder: str,
    ) -> None:
        self.segment_path = segment_path
        self.keyframe_paths = keyframe_paths
        self.video_encoder = video_encoder


def _render_artifacts(
    source: Path,
    asset: AssetCreate,
    output_directory: Path,
) -> _RenderedArtifacts:
    ffmpeg = resolve_video_tool("ffmpeg")
    if ffmpeg is None:
        raise VideoToolingError("missing video dependency: ffmpeg")
    locator = asset.source_locator
    start_ms = _required_int(locator, "start_ms")
    end_ms = _required_int(locator, "end_ms")
    if end_ms <= start_ms:
        raise VideoToolingError("video segment end_ms must be greater than start_ms")

    output_directory.mkdir(parents=True, exist_ok=True)
    encoder, encoder_args = _video_encoder_arguments(ffmpeg)
    segment_path = output_directory / "segment.mp4"
    _run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-i",
            str(source),
            "-ss",
            _seconds(start_ms),
            "-t",
            _seconds(end_ms - start_ms),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            encoder,
            *encoder_args,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(segment_path),
        ],
    )
    if not segment_path.is_file() or segment_path.stat().st_size == 0:
        raise VideoToolingError("FFmpeg did not create a video segment")

    keyframe_paths: list[Path] = []
    for ordinal, summary in enumerate(_representative_summaries(asset), start=1):
        keyframe_path = output_directory / f"keyframe-{ordinal:02d}.jpg"
        _run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-i",
                str(source),
                "-ss",
                _seconds(_required_int(summary, "timestamp_ms")),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(keyframe_path),
            ],
        )
        if not keyframe_path.is_file() or keyframe_path.stat().st_size == 0:
            raise VideoToolingError("FFmpeg did not create a representative keyframe")
        keyframe_paths.append(keyframe_path)

    return _RenderedArtifacts(
        segment_path=segment_path,
        keyframe_paths=keyframe_paths,
        video_encoder=encoder,
    )


def _object_prefix(asset: AssetCreate) -> str:
    return "/".join(
        (
            "derived",
            "video-segments",
            asset.workspace_id,
            asset.source_file_id,
            asset.asset_key,
        )
    )


def _resolve_source(source_file: DiscoveredFile) -> Path:
    source = Path(source_file.path).expanduser().resolve()
    if not source.is_file():
        raise VideoToolingError(f"video source does not exist: {source}")
    return source


def _representative_summaries(asset: AssetCreate) -> list[dict[str, object]]:
    raw = asset.file_info.get("representative_frames")
    if not isinstance(raw, list) or not raw:
        raise VideoToolingError("video asset is missing representative frame metadata")
    summaries: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise VideoToolingError("video representative frame metadata must be objects")
        _required_int(item, "timestamp_ms")
        summaries.append(cast(dict[str, object], item))
    return summaries


def _required_int(value: dict[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool):
        raise VideoToolingError(f"video metadata {key!r} must be an integer")
    try:
        return int(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise VideoToolingError(f"video metadata {key!r} must be an integer") from exc


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1000:.3f}"


@lru_cache(maxsize=4)
def _video_encoder_arguments(ffmpeg: Path) -> tuple[str, tuple[str, ...]]:
    completed = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        capture_output=True,
        check=True,
        text=True,
    )
    encoders = completed.stdout + completed.stderr
    if "libx264" in encoders:
        return "libx264", ("-preset", "veryfast", "-crf", "23")
    if "h264_videotoolbox" in encoders:
        return "h264_videotoolbox", ("-b:v", "4M")
    return "mpeg4", ("-q:v", "4")


def _run_ffmpeg(ffmpeg: Path, arguments: list[str]) -> None:
    try:
        subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "FFmpeg failed").strip()
        raise VideoToolingError(f"FFmpeg media generation failed: {detail[:1000]}") from exc
