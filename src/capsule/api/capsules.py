from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from capsule.search.history import SearchCapsuleNotFoundError, SearchHistoryRepository
from capsule.search.models import (
    SearchCapsuleDetail,
    SearchCapsuleListResponse,
    SearchCapsulePatch,
    SearchResponse,
)
from capsule.search.service import SearchService

router = APIRouter(prefix="/api/v1/search-capsules", tags=["search-capsules"])


def _history(request: Request) -> SearchHistoryRepository:
    history = getattr(request.app.state, "search_history", None)
    if not isinstance(history, SearchHistoryRepository):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "search_capsules_not_ready",
                "message": "Search Capsule service is not ready",
            },
        )
    return history


def _search(request: Request) -> SearchService:
    search = getattr(request.app.state, "search_service", None)
    if not isinstance(search, SearchService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "search_not_ready",
                "message": "Search requires CAPSULE_ARK_API_KEY",
            },
        )
    return search


@router.get("", response_model=SearchCapsuleListResponse)
async def list_search_capsules(
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    created_by: str = Query(default="user_demo", min_length=1, max_length=128),
    favorites_only: bool = False,
) -> SearchCapsuleListResponse:
    history = _history(request)
    return await history.list_capsules(
        workspace_id=workspace_id,
        created_by=created_by,
        favorites_only=favorites_only,
    )


@router.get("/{capsule_id}", response_model=SearchCapsuleDetail)
async def get_search_capsule(
    capsule_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    created_by: str = Query(default="user_demo", min_length=1, max_length=128),
) -> SearchCapsuleDetail:
    history = _history(request)
    try:
        return await history.get_capsule(
            capsule_id=capsule_id,
            workspace_id=workspace_id,
            created_by=created_by,
        )
    except SearchCapsuleNotFoundError as exc:
        raise _not_found(capsule_id) from exc


@router.post("/{capsule_id}/refresh", response_model=SearchResponse)
async def refresh_search_capsule(
    capsule_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    created_by: str = Query(default="user_demo", min_length=1, max_length=128),
) -> SearchResponse:
    history = _history(request)
    search = _search(request)
    try:
        search_request = await history.load_request(
            capsule_id=capsule_id,
            workspace_id=workspace_id,
            created_by=created_by,
        )
        return await search.search(
            search_request,
            existing_capsule_id=capsule_id,
        )
    except SearchCapsuleNotFoundError as exc:
        raise _not_found(capsule_id) from exc


@router.patch("/{capsule_id}", response_model=SearchCapsuleDetail)
async def patch_search_capsule(
    capsule_id: str,
    payload: SearchCapsulePatch,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    created_by: str = Query(default="user_demo", min_length=1, max_length=128),
) -> SearchCapsuleDetail:
    history = _history(request)
    try:
        return await history.set_favorite(
            capsule_id=capsule_id,
            workspace_id=workspace_id,
            created_by=created_by,
            is_favorite=payload.is_favorite,
        )
    except SearchCapsuleNotFoundError as exc:
        raise _not_found(capsule_id) from exc


@router.delete("/{capsule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_search_capsule(
    capsule_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    created_by: str = Query(default="user_demo", min_length=1, max_length=128),
) -> Response:
    history = _history(request)
    try:
        await history.delete_capsule(
            capsule_id=capsule_id,
            workspace_id=workspace_id,
            created_by=created_by,
        )
    except SearchCapsuleNotFoundError as exc:
        raise _not_found(capsule_id) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _not_found(capsule_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "search_capsule_not_found",
            "message": f"Search Capsule {capsule_id!r} was not found",
        },
    )
