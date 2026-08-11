import asyncio
import mimetypes
import os
import platform
import re
import signal
from collections.abc import AsyncIterator
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import unquote, urlencode, urlparse

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from capsule.config import Settings
from capsule.db.repositories import AssetMediaTarget, AssetRepository
from capsule.parsers.video import resolve_video_tool
from capsule.schemas import AssetListResponse, AssetPlayback, AssetViewRecord
from capsule.storage.object_storage import ObjectStorage

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])

_SINGLE_BYTE_RANGE = re.compile(r"^bytes=\d*-\d*$")
_TRANSCODE_GRACE_SECONDS = 3.0


def _repository(request: Request) -> AssetRepository:
    repository = getattr(request.app.state, "asset_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "asset_repository_not_ready", "message": "asset storage is not ready"},
        )
    return cast(AssetRepository, repository)


@router.get("", response_model=AssetListResponse)
async def list_assets(
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    asset_type: str | None = None,
    processing_status: str | None = None,
    source_file_id: str | None = None,
    query: str | None = None,
    asset_id: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AssetListResponse:
    result = await _repository(request).list_asset_views(
        workspace_id=workspace_id,
        asset_type=asset_type,
        processing_status=processing_status,
        source_file_id=source_file_id,
        query=query,
        asset_ids=asset_id,
        limit=limit,
        offset=offset,
    )
    return result.model_copy(
        update={
            "items": [_with_media_urls(request, item) for item in result.items],
        }
    )


@router.post("/clear-all")
async def clear_library() -> None:
    """Retired: broad multi-workspace deletion is intentionally unavailable."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "code": "library_clear_retired",
            "message": "delete one current workspace through /api/v1/workspaces/{workspace_id}",
        },
    )


@router.get("/{asset_id}", response_model=AssetViewRecord)
async def get_asset(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> AssetViewRecord:
    try:
        item = await _repository(request).get_asset_view(
            asset_id=asset_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise _not_found(asset_id) from exc
    return _with_media_urls(request, item)


@router.get("/{asset_id}/preview", name="get_asset_preview")
async def get_asset_preview(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> Response:
    target = await _media_target(request, asset_id=asset_id, workspace_id=workspace_id)
    uri = target.preview_uri
    media_type: str | None = None
    if uri is None and target.asset_type == "image":
        uri = target.derived_file_uri or target.source_storage_uri
        if target.derived_file_uri is None:
            media_type = target.source_mime_type
    if uri is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_preview_not_found", "message": "asset has no preview"},
        )
    return await _serve_uri(request, uri=uri, media_type=media_type)


@router.get("/{asset_id}/thumbnail", name="get_asset_thumbnail")
async def get_asset_thumbnail(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> Response:
    target = await _media_target(request, asset_id=asset_id, workspace_id=workspace_id)
    uri = target.preview_uri
    if uri is None and target.asset_type == "image":
        uri = target.derived_file_uri or target.source_storage_uri
    if uri is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_thumbnail_not_found", "message": "asset has no thumbnail"},
        )

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        storage = _object_storage(request)
        source_content = await storage.download_uri(uri)
        try:
            content = await asyncio.to_thread(_render_thumbnail_content, source_content)
        except (OSError, UnidentifiedImageError):
            return Response(
                content=source_content,
                media_type=_guessed_media_type(uri, fallback="application/octet-stream"),
                headers={"Cache-Control": "public, max-age=86400, immutable"},
            )
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )
    if parsed.scheme != "file":
        return await _serve_uri(request, uri=uri, media_type=None)

    settings = cast(Settings, request.app.state.settings)
    path = await asyncio.to_thread(
        _validated_local_path,
        settings,
        unquote(parsed.path),
    )
    stat = await asyncio.to_thread(path.stat)
    try:
        content = await asyncio.to_thread(
            _render_local_thumbnail,
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
        )
    except (OSError, UnidentifiedImageError):
        return FileResponse(path)
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/{asset_id}/content", name="get_asset_content")
async def get_asset_content(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> Response:
    target = await _media_target(request, asset_id=asset_id, workspace_id=workspace_id)
    uses_derived_media = (
        target.asset_type in {"image", "video_segment"}
        and target.derived_file_uri is not None
    )
    uri = target.derived_file_uri if uses_derived_media else target.source_storage_uri
    if uri is None:  # Narrow the optional derived URI for static type checkers.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return await _serve_uri(
        request,
        uri=uri,
        media_type=None if uses_derived_media else target.source_mime_type,
    )


@router.get("/{asset_id}/playback", name="get_asset_transcoded_playback")
async def get_asset_transcoded_playback(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> StreamingResponse:
    """Transcode one local legacy interval directly to a fragmented MP4 stream.

    This endpoint intentionally refuses object-storage sources: downloading an
    arbitrary S3 object to a worker disk would violate the no-persistent-media
    contract.  A video worker may create derived clips for those sources first.
    """
    target = await _media_target(request, asset_id=asset_id, workspace_id=workspace_id)
    if target.derived_file_uri is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "asset_transcode_not_required",
                "message": "asset already has a derived playable clip",
            },
        )
    parsed = urlparse(target.source_storage_uri)
    if parsed.scheme != "file":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={
                "code": "asset_transcode_remote_source_unsupported",
                "message": (
                    "on-demand transcoding requires a local source; "
                    "remote objects are not downloaded"
                ),
            },
        )
    try:
        item = await _repository(request).get_asset_view(
            asset_id=asset_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise _not_found(asset_id) from exc
    start_ms, end_ms = _video_interval(item.source_locator)
    if start_ms is None or end_ms is None or end_ms <= start_ms:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "asset_transcode_interval_missing",
                "message": "on-demand transcoding requires a valid video start_ms/end_ms interval",
            },
        )
    settings = cast(Settings, request.app.state.settings)
    source = await asyncio.to_thread(_validated_local_path, settings, unquote(parsed.path))
    ffmpeg = resolve_video_tool("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "asset_transcode_unavailable",
                "message": "FFmpeg is unavailable on this API worker",
            },
        )
    return StreamingResponse(
        _limited_transcoded_interval_stream(
            semaphore=cast(asyncio.Semaphore, request.app.state.video_transcode_semaphore),
            ffmpeg=ffmpeg,
            source=source,
            start_ms=start_ms,
            end_ms=end_ms,
        ),
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"},
    )


async def _media_target(
    request: Request,
    *,
    asset_id: str,
    workspace_id: str,
) -> AssetMediaTarget:
    try:
        return await _repository(request).get_asset_media(
            asset_id=asset_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise _not_found(asset_id) from exc


async def _serve_uri(
    request: Request,
    *,
    uri: str,
    media_type: str | None,
) -> Response:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        settings = cast(Settings, request.app.state.settings)
        path = await asyncio.to_thread(
            _validated_local_path,
            settings,
            unquote(parsed.path),
        )
        return await _serve_local_uri(request, path=path, media_type=media_type)
    if parsed.scheme == "s3":
        storage = _object_storage(request)
        byte_range = request.headers.get("range")
        if byte_range is not None and not _SINGLE_BYTE_RANGE.fullmatch(byte_range):
            return Response(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
        try:
            payload = await storage.download_uri_response(uri, byte_range=byte_range)
        except ClientError as exc:
            response_status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if response_status == status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE:
                return Response(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)
            raise
        headers = {
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=86400, immutable",
        }
        if payload.content_range:
            headers["Content-Range"] = payload.content_range
        if payload.etag:
            headers["ETag"] = payload.etag
        return Response(
            content=payload.content,
            status_code=(
                status.HTTP_206_PARTIAL_CONTENT
                if payload.content_range
                else status.HTTP_200_OK
            ),
            media_type=(
                media_type
                or payload.content_type
                or _guessed_media_type(uri, fallback="application/octet-stream")
            ),
            headers=headers,
        )
    if parsed.scheme in {"http", "https"}:
        return RedirectResponse(uri)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": "unsupported_asset_media_uri", "message": "unsupported asset media URI"},
    )


async def _serve_local_uri(
    request: Request,
    *,
    path: Path,
    media_type: str | None,
) -> Response:
    """Serve local sources with an explicit single-byte-range implementation."""
    byte_range = request.headers.get("range")
    resolved_media_type = media_type or _guessed_media_type(
        path.as_uri(),
        fallback="application/octet-stream",
    )
    if byte_range is None:
        return FileResponse(path, media_type=resolved_media_type)
    size = (await asyncio.to_thread(path.stat)).st_size
    selected = _parse_byte_range(byte_range, size)
    if selected is None:
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )
    start, end = selected
    content = await asyncio.to_thread(_read_local_range, path, start, end)
    return Response(
        content=content,
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type=resolved_media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )


def _parse_byte_range(value: str, size: int) -> tuple[int, int] | None:
    if size < 1 or not _SINGLE_BYTE_RANGE.fullmatch(value):
        return None
    start_text, _, end_text = value.removeprefix("bytes=").partition("-")
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length < 1:
                return None
            return max(0, size - suffix_length), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    if start < 0 or start >= size or end < start:
        return None
    return start, min(end, size - 1)


def _read_local_range(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as source:
        source.seek(start)
        return source.read(end - start + 1)


async def _transcoded_interval_stream(
    *,
    ffmpeg: Path,
    source: Path,
    start_ms: int,
    end_ms: int,
) -> AsyncIterator[bytes]:
    """Yield a bounded local interval without creating a media file on disk."""
    video_encoder_args = (
        ["-c:v", "h264_videotoolbox", "-b:v", "4M"]
        if platform.system() == "Darwin"
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    )
    process = await asyncio.create_subprocess_exec(
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        _ffmpeg_seconds(start_ms),
        "-i",
        str(source),
        "-t",
        _ffmpeg_seconds(end_ms - start_ms),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        *video_encoder_args,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "-f",
        "mp4",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:  # pragma: no cover - asyncio invariant
        await _terminate_transcode_process(process)
        raise RuntimeError("FFmpeg did not expose transcoding pipes")
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        while chunk := await process.stdout.read(64 * 1024):
            yield chunk
        return_code = await process.wait()
        stderr = await stderr_task
        if return_code:
            detail = stderr.decode(errors="replace").strip() or "FFmpeg transcoding failed"
            raise RuntimeError(detail[:1_000])
    finally:
        if process.returncode is None:
            await _terminate_transcode_process(process)
        if not stderr_task.done():
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)


async def _limited_transcoded_interval_stream(
    *,
    semaphore: asyncio.Semaphore,
    ffmpeg: Path,
    source: Path,
    start_ms: int,
    end_ms: int,
) -> AsyncIterator[bytes]:
    """Limit concurrent encoders while preserving streaming backpressure."""
    async with semaphore:
        async for chunk in _transcoded_interval_stream(
            ffmpeg=ffmpeg,
            source=source,
            start_ms=start_ms,
            end_ms=end_ms,
        ):
            yield chunk


async def _terminate_transcode_process(process: asyncio.subprocess.Process) -> None:
    """Reap an interrupted FFmpeg process group before ending the HTTP stream."""
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=_TRANSCODE_GRACE_SECONDS)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def _ffmpeg_seconds(milliseconds: int) -> str:
    return f"{milliseconds / 1_000:.3f}"


def _validated_local_path(settings: Settings, raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    allowed_roots = (
        settings.import_root.expanduser().resolve(),
        settings.document_media_root.expanduser().resolve(),
        *(root.expanduser().resolve() for root in settings.video_source_roots),
    )
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "asset_media_outside_allowed_roots",
                "message": "local media is outside the configured media roots",
            },
        )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_media_not_found", "message": "asset media file is missing"},
        )
    return path


def _object_storage(request: Request) -> ObjectStorage:
    storage = getattr(request.app.state, "object_storage", None)
    if not isinstance(storage, ObjectStorage):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "object_storage_not_ready",
                "message": "object storage is not ready",
            },
        )
    return storage


def _guessed_media_type(uri: str, *, fallback: str) -> str:
    return mimetypes.guess_type(urlparse(uri).path)[0] or fallback


@lru_cache(maxsize=256)
def _render_local_thumbnail(
    path_value: str,
    _modified_at_ns: int,
    _source_size: int,
) -> bytes:
    with open(path_value, "rb") as source:
        return _render_thumbnail_content(source.read())


def _render_thumbnail_content(content: bytes) -> bytes:
    with Image.open(BytesIO(content)) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail((480, 480), Image.Resampling.LANCZOS)
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            flattened = Image.new("RGB", rgba.size, "white")
            flattened.paste(rgba, mask=rgba.getchannel("A"))
            image = flattened
        elif image.mode != "RGB":
            image = image.convert("RGB")
        output = BytesIO()
        image.save(output, format="JPEG", quality=76, optimize=True, progressive=True)
        return output.getvalue()


def _with_media_urls(request: Request, item: AssetViewRecord) -> AssetViewRecord:
    query = {"workspace_id": item.workspace_id}
    query_string = urlencode(query)
    content_url = (
        f"{request.url_for('get_asset_content', asset_id=item.asset_id).path}"
        f"?{query_string}"
    )
    preview_url = (
        f"{request.url_for('get_asset_thumbnail', asset_id=item.asset_id).path}"
        f"?{query_string}"
        if item.asset_type.value == "image"
        or (item.asset_type.value == "video_segment" and item.preview_uri is not None)
        else None
    )
    playback = _playback_description(
        request,
        item,
        content_url=content_url,
    )
    return item.model_copy(
        update={
            "preview_url": preview_url,
            "content_url": content_url,
            "playback": playback,
        }
    )


def _playback_description(
    request: Request,
    item: AssetViewRecord,
    *,
    content_url: str,
) -> AssetPlayback | None:
    if item.asset_type.value != "video_segment":
        return None
    start_ms, end_ms = _video_interval(item.source_locator)
    derived = item.derived_file_uri
    if derived is not None:
        mime_type = _guessed_media_type(derived, fallback="video/mp4")
        return AssetPlayback(
            mode="derived_clip",
            url=content_url,
            fallback_url=None,
            mime_type=mime_type,
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=_duration_ms(start_ms, end_ms),
            browser_compatible=_browser_compatible_video_mime(mime_type),
        )
    mime_type = item.source_file.mime_type or _guessed_media_type(
        item.file_name,
        fallback="application/octet-stream",
    )
    browser_compatible = _browser_compatible_video_mime(mime_type)
    transcode_url = _local_transcode_fallback_url(request, item)
    if not browser_compatible and transcode_url is not None:
        return AssetPlayback(
            mode="transcoded_stream",
            url=transcode_url,
            mime_type="video/mp4",
            start_ms=start_ms,
            end_ms=end_ms,
            duration_ms=_duration_ms(start_ms, end_ms),
            browser_compatible=True,
        )
    return AssetPlayback(
        mode="source_range",
        url=content_url,
        fallback_url=transcode_url,
        mime_type=mime_type,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=_duration_ms(start_ms, end_ms),
        browser_compatible=browser_compatible,
    )


def _local_transcode_fallback_url(request: Request, item: AssetViewRecord) -> str | None:
    """Expose a one-shot fallback only when the source is locally trusted."""
    if item.source_storage_uri is None or urlparse(item.source_storage_uri).scheme != "file":
        return None
    settings = cast(Settings, request.app.state.settings)
    try:
        _validated_local_path(settings, unquote(urlparse(item.source_storage_uri).path))
    except HTTPException:
        return None
    query_string = urlencode({"workspace_id": item.workspace_id})
    return (
        f"{request.url_for('get_asset_transcoded_playback', asset_id=item.asset_id).path}"
        f"?{query_string}"
    )


def _video_interval(locator: dict[str, object]) -> tuple[int | None, int | None]:
    """Read current millisecond locators and legacy *_time/seconds variants."""
    start = _first_number(locator, "start_ms", "start_time_ms")
    end = _first_number(locator, "end_ms", "end_time_ms")
    if start is None:
        seconds = _first_number(locator, "start_seconds", "start_time_seconds")
        start = None if seconds is None else round(seconds * 1_000)
    if end is None:
        seconds = _first_number(locator, "end_seconds", "end_time_seconds")
        end = None if seconds is None else round(seconds * 1_000)
    if start is not None and start < 0:
        start = None
    if end is not None and (end < 0 or (start is not None and end < start)):
        end = None
    return (
        None if start is None else round(start),
        None if end is None else round(end),
    )


def _first_number(locator: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = locator.get(key)
        if isinstance(value, bool):
            continue
        try:
            return float(cast(float | int | str, value))
        except (TypeError, ValueError):
            continue
    return None


def _duration_ms(start_ms: int | None, end_ms: int | None) -> int | None:
    return end_ms - start_ms if start_ms is not None and end_ms is not None else None


def _browser_compatible_video_mime(mime_type: str) -> bool:
    return mime_type.lower().split(";", maxsplit=1)[0] in {
        "video/mp4",
        "video/webm",
        "video/ogg",
    }


def _not_found(asset_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "asset_not_found",
            "message": f"Asset {asset_id!r} was not found",
        },
    )
