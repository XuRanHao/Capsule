from typing import Any, Protocol, cast

from fastapi import APIRouter, HTTPException, Query, Request, status


class RelationGraphBuilder(Protocol):
    async def build(
        self,
        *,
        workspace_id: str,
        force_understanding: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]: ...


router = APIRouter(prefix="/api/v1/relation-graphs", tags=["relation-graphs"])


def _service(request: Request) -> RelationGraphBuilder:
    service = getattr(request.app.state, "relation_graph_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "relation_graph_service_not_ready",
                "message": "relationship graph construction is not ready",
            },
        )
    return cast(RelationGraphBuilder, service)


@router.post("/build")
async def build_relation_graph(
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    force_understanding: bool = False,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    return await _service(request).build(
        workspace_id=workspace_id,
        force_understanding=force_understanding,
        force_rebuild=force_rebuild,
    )
