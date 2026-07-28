from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError

from capsule.search.models import QueryImageUploadResponse, SearchRequest, SearchResponse
from capsule.search.query_embedding import QueryEmbeddingError
from capsule.search.service import SearchService, SearchUnavailableError
from capsule.search.uploads import QueryImageService

router = APIRouter(prefix="/api/v1", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    if not isinstance(service, SearchService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "search_not_ready", "message": "search service is not ready"},
        )
    return service


def get_query_image_service(request: Request) -> QueryImageService:
    service = getattr(request.app.state, "query_image_service", None)
    if not isinstance(service, QueryImageService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "query_image_upload_not_ready",
                "message": "upload service is not ready",
            },
        )
    return service


@router.post("/search", response_model=SearchResponse)
async def search_assets(payload: SearchRequest, request: Request) -> SearchResponse:
    service = get_search_service(request)
    try:
        return await service.search(payload)
    except QueryEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "query_embedding_failed", "message": str(exc)},
        ) from exc
    except SearchUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "search_unavailable", "message": str(exc)},
        ) from exc


@router.post("/query-images", response_model=QueryImageUploadResponse)
async def upload_query_image(
    request: Request,
    workspace_id: Annotated[str, Form(min_length=1, max_length=64)],
    file: Annotated[UploadFile, File()],
) -> QueryImageUploadResponse:
    service = get_query_image_service(request)
    max_bytes = request.app.state.settings.query_image_max_bytes
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "image_too_large", "message": "query image exceeds size limit"},
        )
    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "not_an_image", "message": "only image uploads are supported"},
        )
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_image", "message": "uploaded content is not a valid image"},
        ) from exc
    upload_id, image_url = await service.upload(
        workspace_id=workspace_id,
        content=content,
        content_type=content_type,
        original_file_name=file.filename or "query-image",
    )
    return QueryImageUploadResponse(upload_id=upload_id, image_url=image_url)
