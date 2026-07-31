"""Persist the playable artifacts belonging to analyzed video Segment Assets."""

import asyncio
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, Field

from capsule.enums import AssetType
from capsule.parsers.video import VideoToolingError, resolve_video_tool
from capsule.pipeline.video_upload_queue import (
    InMemoryUploadQueue,
    RedisStreamUploadQueue,
    UploadQueue,
    UploadQueueItem,
)
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


AssetPersistedCallback = Callable[[AssetCreate, int], Awaitable[str]]
AssetCommittedCallback = Callable[[str], Awaitable[None]]
AssetGenerationValidator = Callable[[AssetCreate], Awaitable[None]]
_DEFAULT_SPOOL_ROOT = Path(tempfile.gettempdir()) / "capsule-video-spool"


class ObsoleteVideoUploadError(RuntimeError):
    """A recovered manifest can no longer publish to its stable object keys."""


class _UploadManifest(BaseModel):
    manifest_id: str
    asset: AssetCreate
    segment_path: str
    keyframe_paths: list[str] = Field(min_length=1)
    video_encoder: str
    spool_bytes: int = Field(gt=0)
    generation_asset_count: int = Field(gt=0)
    attempt: int = Field(default=0, ge=0)


class _DiskSpool:
    """Bound ready bundles while allowing FFmpeg slots to release immediately."""

    def __init__(self, root: Path, *, max_items: int, max_bytes: int) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._max_items = max_items
        self._slots = asyncio.Semaphore(max_items)
        self._max_bytes = max_bytes
        self._ready_bytes = 0
        self._condition = asyncio.Condition()

    async def allocate(self, asset: AssetCreate) -> Path:
        await self._slots.acquire()
        directory = self.root / (
            f"{asset.source_file_id}-{asset.generation}-{asset.asset_key[:12]}-{uuid.uuid4().hex}"
        )
        return directory

    async def commit(self, manifest: _UploadManifest, directory: Path) -> Path:
        async with self._condition:
            while self._ready_bytes and self._ready_bytes + manifest.spool_bytes > self._max_bytes:
                await self._condition.wait()
            self._ready_bytes += manifest.spool_bytes
        manifest_path = directory / "manifest.json"
        try:
            self.write_manifest(manifest_path, manifest)
        except BaseException:
            async with self._condition:
                self._ready_bytes = max(0, self._ready_bytes - manifest.spool_bytes)
                self._condition.notify_all()
            raise
        return manifest_path

    def write_manifest(self, manifest_path: Path, manifest: _UploadManifest) -> None:
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(manifest.model_dump_json(), encoding="utf-8")
        temporary_manifest.replace(manifest_path)

    def recover_manifests(self) -> list[Path]:
        manifests: list[Path] = []
        ready_bytes = 0
        for temporary_manifest in sorted(self.root.glob("*/manifest.json.tmp")):
            try:
                _read_manifest(temporary_manifest)
            except (OSError, ValueError):
                continue
            temporary_manifest.replace(temporary_manifest.with_suffix(""))
        for manifest_path in sorted(self.root.glob("*/manifest.json")):
            try:
                manifest = _read_manifest(manifest_path)
            except (OSError, ValueError):
                continue
            manifests.append(manifest_path)
            ready_bytes += manifest.spool_bytes
        self._ready_bytes = ready_bytes
        self._slots = asyncio.Semaphore(max(0, self._max_items - len(manifests)))
        return manifests

    def queue_path(self, manifest_path: Path) -> str:
        return manifest_path.resolve().relative_to(self.root).as_posix()

    def resolve_queue_path(self, value: str) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        )
        resolved.relative_to(self.root)
        return resolved

    async def release(self, manifest_path: Path, spool_bytes: int) -> None:
        await asyncio.to_thread(shutil.rmtree, manifest_path.parent, True)
        async with self._condition:
            self._ready_bytes = max(0, self._ready_bytes - spool_bytes)
            self._condition.notify_all()
        self._slots.release()

    async def abort(self, directory: Path) -> None:
        await asyncio.to_thread(shutil.rmtree, directory, True)
        self._slots.release()


class VideoDerivedMediaWriter:
    """Create a playable clip, cover image, and keyframe images for video Assets.

    The database Asset stays a logical time range. These files make that range
    playable and can later be supplied unchanged to video embedding calls.
    """

    def __init__(
        self,
        storage: VideoArtifactStorage,
        *,
        concurrency: int = 1,
        upload_concurrency: int = 4,
        spool_root: Path = _DEFAULT_SPOOL_ROOT,
        spool_max_items: int = 32,
        spool_max_bytes: int = 4 * 1024 * 1024 * 1024,
        queue_backend: Literal["memory", "redis"] = "memory",
        redis_url: str = "redis://localhost:6379/0",
        redis_stream: str = "capsule:video-uploads",
        redis_group: str = "capsule-video-uploaders",
        redis_claim_idle_ms: int = 30_000,
        max_upload_attempts: int = 4,
        retry_base_seconds: float = 0.5,
        upload_queue: UploadQueue | None = None,
        on_asset_persisted: AssetPersistedCallback | None = None,
        validate_asset_generation: AssetGenerationValidator | None = None,
    ) -> None:
        if min(concurrency, upload_concurrency, spool_max_items, spool_max_bytes) < 1:
            raise ValueError("video pipeline concurrency and spool limits must be positive")
        if max_upload_attempts < 1 or retry_base_seconds < 0:
            raise ValueError("video upload retry settings are invalid")
        self._storage = storage
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._upload_concurrency = upload_concurrency
        self._spool = _DiskSpool(
            spool_root,
            max_items=spool_max_items,
            max_bytes=spool_max_bytes,
        )
        self._queue = upload_queue or (
            InMemoryUploadQueue(maxsize=spool_max_items)
            if queue_backend == "memory"
            else RedisStreamUploadQueue(
                redis_url=redis_url,
                stream=redis_stream,
                group=redis_group,
                consumer=f"video-{uuid.uuid4().hex}",
                claim_idle_ms=redis_claim_idle_ms,
            )
        )
        self._max_upload_attempts = max_upload_attempts
        self._retry_base_seconds = retry_base_seconds
        self._on_asset_persisted = on_asset_persisted
        self._validate_asset_generation = validate_asset_generation
        self._bucket_lock = asyncio.Lock()
        self._bucket_ready = False
        self._start_lock = asyncio.Lock()
        self._workers: list[asyncio.Task[None]] = []
        self._pending: dict[
            str,
            tuple[asyncio.Future[AssetCreate], AssetCommittedCallback | None],
        ] = {}
        self._recovered_paths: set[str] = set()
        self._recovered_commits: list[tuple[str, str]] = []

    async def persist(
        self,
        *,
        source_file: DiscoveredFile,
        assets: Sequence[AssetCreate],
        on_asset_committed: AssetCommittedCallback | None = None,
    ) -> list[AssetCreate]:
        video_assets = [asset for asset in assets if asset.asset_type is AssetType.VIDEO_SEGMENT]
        if not video_assets:
            return list(assets)

        source = await asyncio.to_thread(_resolve_source, source_file)
        source_has_audio = await asyncio.to_thread(_source_has_audio, source)
        await self._ensure_bucket()
        await self._start()
        persisted = await self._persist_many(
            source=source,
            source_has_audio=source_has_audio,
            assets=video_assets,
            on_asset_committed=on_asset_committed,
        )
        updated = {asset.asset_key: asset for asset in persisted}
        return [updated.get(asset.asset_key, asset) for asset in assets]

    async def _persist_many(
        self,
        *,
        source: Path,
        source_has_audio: bool,
        assets: list[AssetCreate],
        on_asset_committed: AssetCommittedCallback | None,
    ) -> list[AssetCreate]:
        persisted: list[AssetCreate | None] = [None] * len(assets)
        workers = [
            asyncio.create_task(
                self._render_and_enqueue(
                    source=source,
                    source_has_audio=source_has_audio,
                    asset=asset,
                    generation_asset_count=len(assets),
                    on_asset_committed=on_asset_committed,
                )
            )
            for asset in assets
        ]
        try:
            results = await asyncio.gather(*workers)
            for index, asset in enumerate(results):
                persisted[index] = asset
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        if any(asset is None for asset in persisted):
            raise RuntimeError("video media persistence ended before all assets completed")
        return [asset for asset in persisted if asset is not None]

    async def close(self) -> None:
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await self._queue.close()

    async def start(self) -> list[tuple[str, str]]:
        """Start consumers and republish complete manifests left in the spool."""
        return await self._start()

    async def _start(self) -> list[tuple[str, str]]:
        if self._workers:
            return []
        async with self._start_lock:
            if self._workers:
                return []
            await self._queue.start()
            recovered_manifests = self._spool.recover_manifests()
            if recovered_manifests:
                await self._ensure_bucket()
            self._workers = [
                asyncio.create_task(self._upload_worker()) for _ in range(self._upload_concurrency)
            ]
            recovered_futures: list[asyncio.Future[AssetCreate]] = []
            for manifest_path in recovered_manifests:
                manifest = _read_manifest(manifest_path)
                future: asyncio.Future[AssetCreate] = asyncio.get_running_loop().create_future()
                self._pending[str(manifest_path)] = (future, None)
                self._recovered_paths.add(str(manifest_path))
                recovered_futures.append(future)
                try:
                    await self._publish(
                        UploadQueueItem(
                            manifest_path=self._spool.queue_path(manifest_path),
                            attempt=manifest.attempt,
                        )
                    )
                except Exception as exc:
                    self._pending.pop(str(manifest_path), None)
                    self._recovered_paths.discard(str(manifest_path))
                    future.set_exception(exc)
            if recovered_futures:
                await asyncio.gather(*recovered_futures, return_exceptions=True)
            recovered_commits, self._recovered_commits = self._recovered_commits, []
            return recovered_commits

    async def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        async with self._bucket_lock:
            if not self._bucket_ready:
                await self._storage.ensure_bucket()
                self._bucket_ready = True

    async def _render_and_enqueue(
        self,
        *,
        source: Path,
        source_has_audio: bool,
        asset: AssetCreate,
        generation_asset_count: int,
        on_asset_committed: AssetCommittedCallback | None,
    ) -> AssetCreate:
        output_directory = await self._spool.allocate(asset)
        manifest_path: Path | None = None
        try:
            async with self._semaphore:
                artifacts = await asyncio.to_thread(
                    _render_artifacts,
                    source,
                    asset,
                    output_directory,
                    source_has_audio,
                )
            spool_bytes = sum(
                path.stat().st_size for path in [artifacts.segment_path, *artifacts.keyframe_paths]
            )
            manifest_id = uuid.uuid4().hex
            manifest = _UploadManifest(
                manifest_id=manifest_id,
                asset=asset,
                segment_path=artifacts.segment_path.name,
                keyframe_paths=[path.name for path in artifacts.keyframe_paths],
                video_encoder=artifacts.video_encoder,
                spool_bytes=spool_bytes,
                generation_asset_count=generation_asset_count,
            )
            manifest_path = await self._spool.commit(manifest, output_directory)
            future: asyncio.Future[AssetCreate] = asyncio.get_running_loop().create_future()
            self._pending[str(manifest_path)] = (future, on_asset_committed)
            await self._publish(
                UploadQueueItem(manifest_path=self._spool.queue_path(manifest_path))
            )
            return await future
        except BaseException:
            if manifest_path is None:
                await self._spool.abort(output_directory)
            else:
                # Once a complete manifest exists, the queue owns the bundle.
                # Keep it recoverable even if its producing request is cancelled
                # or publishing had an ambiguous network outcome.
                self._pending.pop(str(manifest_path), None)
            raise

    async def _publish(self, item: UploadQueueItem) -> None:
        for attempt in range(self._max_upload_attempts):
            try:
                await self._queue.publish(item)
                return
            except Exception:
                if attempt + 1 >= self._max_upload_attempts:
                    raise
                if self._retry_base_seconds:
                    await asyncio.sleep(min(30.0, self._retry_base_seconds * (2**attempt)))

    async def _upload_worker(self) -> None:
        while True:
            try:
                delivery = await self._queue.receive()
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(max(0.1, self._retry_base_seconds))
                continue
            try:
                manifest_path = self._spool.resolve_queue_path(delivery.item.manifest_path)
            except ValueError:
                try:
                    await self._queue.acknowledge(delivery)
                except Exception:
                    await asyncio.sleep(max(0.1, self._retry_base_seconds))
                continue
            manifest: _UploadManifest | None = None
            persisted: AssetCreate | None = None
            try:
                manifest = _read_manifest(manifest_path)
                if self._validate_asset_generation is not None:
                    await self._validate_asset_generation(manifest.asset)
                persisted = await self._upload_manifest(manifest, manifest_path)
                asset_id = persisted.asset_id
                if self._on_asset_persisted is not None:
                    asset_id = await self._on_asset_persisted(
                        persisted,
                        manifest.generation_asset_count,
                    )
                if str(manifest_path) in self._recovered_paths:
                    self._recovered_paths.discard(str(manifest_path))
                    self._recovered_commits.append((persisted.workspace_id, asset_id))
                pending = self._pending.get(str(manifest_path))
                if pending is not None and pending[1] is not None:
                    await pending[1](asset_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                effective_attempt = max(
                    delivery.item.attempt,
                    manifest.attempt if manifest is not None else 0,
                )
                if (
                    not isinstance(exc, ObsoleteVideoUploadError)
                    and effective_attempt + 1 < self._max_upload_attempts
                ):
                    if self._retry_base_seconds:
                        await asyncio.sleep(
                            min(
                                30.0,
                                self._retry_base_seconds * (2**effective_attempt),
                            )
                        )
                    try:
                        if manifest is not None:
                            manifest = manifest.model_copy(
                                update={"attempt": effective_attempt + 1}
                            )
                            self._spool.write_manifest(manifest_path, manifest)
                        await self._queue.retry(
                            delivery,
                            next_attempt=effective_attempt + 1,
                        )
                    except Exception:
                        await asyncio.sleep(max(0.1, self._retry_base_seconds))
                    continue
                try:
                    await self._queue.acknowledge(delivery)
                except Exception:
                    await asyncio.sleep(max(0.1, self._retry_base_seconds))
                    continue
                self._recovered_paths.discard(str(manifest_path))
                if manifest is not None:
                    await self._spool.release(manifest_path, manifest.spool_bytes)
                else:
                    await self._spool.abort(manifest_path.parent)
                pending = self._pending.pop(str(manifest_path), None)
                if pending is not None and not pending[0].done():
                    pending[0].set_exception(exc)
                continue

            try:
                await self._queue.acknowledge(delivery)
            except Exception:
                # DB state is already durable and the active producer has been
                # released. Keep the spool bundle for a later XAUTOCLAIM pass.
                pending = self._pending.pop(str(manifest_path), None)
                if pending is not None and not pending[0].done() and persisted is not None:
                    pending[0].set_result(persisted)
                await asyncio.sleep(max(0.1, self._retry_base_seconds))
                continue
            assert manifest is not None
            assert persisted is not None
            await self._spool.release(manifest_path, manifest.spool_bytes)
            pending = self._pending.pop(str(manifest_path), None)
            if pending is not None and not pending[0].done():
                pending[0].set_result(persisted)

    async def _upload_manifest(
        self,
        manifest: _UploadManifest,
        manifest_path: Path,
    ) -> AssetCreate:
        asset = manifest.asset
        prefix = _object_prefix(asset)
        bundle = manifest_path.parent
        segment_path = bundle / manifest.segment_path
        keyframe_paths = [bundle / path for path in manifest.keyframe_paths]
        uploads = await asyncio.gather(
            self._storage.upload_file(
                segment_path,
                f"{prefix}/segment.mp4",
                content_type="video/mp4",
            ),
            self._storage.upload_file(
                keyframe_paths[0],
                f"{prefix}/preview.jpg",
                content_type="image/jpeg",
            ),
            *(
                self._storage.upload_file(
                    frame_path,
                    f"{prefix}/keyframes/{ordinal:02d}.jpg",
                    content_type="image/jpeg",
                )
                for ordinal, frame_path in enumerate(keyframe_paths, start=1)
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
            "video_codec": manifest.video_encoder,
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


def _read_manifest(path: Path) -> _UploadManifest:
    return _UploadManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _render_artifacts(
    source: Path,
    asset: AssetCreate,
    output_directory: Path,
    source_has_audio: bool,
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
    representative_summaries = _representative_summaries(asset)
    fps = _required_float(asset.file_info, "fps")
    representative_frame_indices = [
        round((_required_int(summary, "timestamp_ms") - start_ms) * fps / 1000)
        for summary in representative_summaries
    ]
    if any(frame_index < 0 for frame_index in representative_frame_indices):
        raise VideoToolingError("video representative frame precedes its Segment")
    select_expression = "+".join(
        f"eq(n\\,{frame_index})" for frame_index in representative_frame_indices
    )
    filters = [
        "[0:v:0]split=2[segment_video][keyframe_source]",
        f"[keyframe_source]select='{select_expression}'[keyframes]",
    ]
    if source_has_audio:
        filters.append(
            f"[1:a:0]atrim=start={_seconds(start_ms)}:"
            f"duration={_seconds(end_ms - start_ms)},"
            "asetpts=PTS-STARTPTS[segment_audio]"
        )
    segment_arguments = [
        "-y",
        "-ss",
        _seconds(start_ms),
        "-i",
        str(source),
    ]
    if source_has_audio:
        segment_arguments.extend(
            [
                "-i",
                str(source),
            ]
        )
    segment_arguments.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[segment_video]",
        ]
    )
    if source_has_audio:
        segment_arguments.extend(["-map", "[segment_audio]"])
    segment_arguments.extend(
        [
            "-t",
            _seconds(end_ms - start_ms),
            "-c:v",
            encoder,
            *encoder_args,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(segment_path),
            "-map",
            "[keyframes]",
            "-t",
            _seconds(end_ms - start_ms),
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            "-start_number",
            "1",
            str(output_directory / "keyframe-%02d.jpg"),
        ]
    )
    _run_ffmpeg(
        ffmpeg,
        segment_arguments,
    )
    if not segment_path.is_file() or segment_path.stat().st_size == 0:
        raise VideoToolingError("FFmpeg did not create a video segment")

    keyframe_paths = [
        output_directory / f"keyframe-{ordinal:02d}.jpg"
        for ordinal in range(1, len(representative_summaries) + 1)
    ]
    for keyframe_path in keyframe_paths:
        if not keyframe_path.is_file() or keyframe_path.stat().st_size == 0:
            raise VideoToolingError("FFmpeg did not create a representative keyframe")

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


def _source_has_audio(source: Path) -> bool:
    ffprobe = resolve_video_tool("ffprobe")
    if ffprobe is None:
        raise VideoToolingError("missing video dependency: ffprobe")
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode:
        raise VideoToolingError(
            completed.stderr.strip() or f"ffprobe failed while inspecting audio: {source}"
        )
    return bool(completed.stdout.strip())


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


def _required_float(value: dict[str, object], key: str) -> float:
    raw = value.get(key)
    if isinstance(raw, bool):
        raise VideoToolingError(f"video metadata {key!r} must be a number")
    try:
        result = float(cast(int | str | float, raw))
    except (TypeError, ValueError) as exc:
        raise VideoToolingError(f"video metadata {key!r} must be a number") from exc
    if result <= 0:
        raise VideoToolingError(f"video metadata {key!r} must be positive")
    return result


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
