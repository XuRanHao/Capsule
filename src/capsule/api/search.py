from fastapi import APIRouter, HTTPException, Request, status

from capsule.search.models import SearchRequest, SearchResponse
from capsule.search.query_embedding import QueryEmbeddingError
from capsule.search.service import SearchService, SearchUnavailableError

router = APIRouter(prefix="/api/v1", tags=["search"])


def get_search_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "search_service", None)
    if not isinstance(service, SearchService):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "search_not_ready", "message": "search service is not ready"},
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
