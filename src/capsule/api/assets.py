import asyncio
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from capsule.config import Settings
from capsule.db.repositories import AssetMediaTarget, AssetRepository, LibraryClearBusyError
from capsule.pipeline.workspace_clear import LibraryClearService
from capsule.schemas import AssetListResponse, AssetViewRecord, LibraryClearResult
from capsule.storage.object_storage import ObjectStorage

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])

_LIBRARY_CLEAR_CONFIRMATION = "CLEAR ALL DATA"


class LibraryClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation: str = Field(min_length=1, max_length=64)


def _repository(request: Request) -> AssetRepository:
    repository = getattr(request.app.state, "asset_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "asset_repository_not_ready", "message": "asset storage is not ready"},
        )
    return cast(AssetRepository, repository)


def _library_clear_service(request: Request) -> LibraryClearService:
    service = getattr(request.app.state, "library_clear_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "library_clear_not_ready",
                "message": "asset library cleanup is not ready",
            },
        )
    return cast(LibraryClearService, service)


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


@router.post("/clear-all", response_model=LibraryClearResult)
async def clear_library(
    payload: LibraryClearRequest,
    request: Request,
) -> LibraryClearResult:
    """Irreversibly clear all Asset-library data across every workspace."""
    if payload.confirmation != _LIBRARY_CLEAR_CONFIRMATION:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "library_clear_confirmation_required",
                "message": "an explicit library-clear confirmation is required",
            },
        )
    try:
        return await _library_clear_service(request).clear_all()
    except LibraryClearBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "library_has_active_jobs", "message": str(exc)},
        ) from exc


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
        uri = target.source_storage_uri
        media_type = target.source_mime_type
    if uri is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_preview_not_found", "message": "asset has no preview"},
        )
    return await _serve_uri(request, uri=uri, media_type=media_type)


@router.get("/{asset_id}/content", name="get_asset_content")
async def get_asset_content(
    asset_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> Response:
    target = await _media_target(request, asset_id=asset_id, workspace_id=workspace_id)
    uri = (
        target.derived_file_uri
        if target.asset_type == "video_segment" and target.derived_file_uri
        else target.source_storage_uri
    )
    return await _serve_uri(request, uri=uri, media_type=target.source_mime_type)


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
        return FileResponse(path, media_type=media_type)
    if parsed.scheme == "s3":
        storage = getattr(request.app.state, "object_storage", None)
        if not isinstance(storage, ObjectStorage):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "object_storage_not_ready",
                    "message": "object storage is not ready",
                },
            )
        return RedirectResponse(await storage.presigned_get_uri(uri))
    if parsed.scheme in {"http", "https"}:
        return RedirectResponse(uri)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": "unsupported_asset_media_uri", "message": "unsupported asset media URI"},
    )


def _validated_local_path(settings: Settings, raw_path: str) -> Path:
    path = Path(raw_path).resolve()
    import_root = settings.import_root.expanduser().resolve()
    try:
        path.relative_to(import_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "asset_media_outside_import_root",
                "message": "local media is outside the browser import root",
            },
        ) from exc
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_media_not_found", "message": "asset media file is missing"},
        )
    return path


def _with_media_urls(request: Request, item: AssetViewRecord) -> AssetViewRecord:
    query = {"workspace_id": item.workspace_id}
    content_url = str(
        request.url_for("get_asset_content", asset_id=item.asset_id).include_query_params(**query)
    )
    preview_url = (
        str(
            request.url_for("get_asset_preview", asset_id=item.asset_id).include_query_params(
                **query
            )
        )
        if item.asset_type.value in {"image", "video_segment"}
        else None
    )
    return item.model_copy(
        update={
            "preview_url": preview_url,
            "content_url": content_url,
        }
    )


def _not_found(asset_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "asset_not_found",
            "message": f"Asset {asset_id!r} was not found",
        },
    )
