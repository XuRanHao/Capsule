"""HTTP endpoints for creating narrative graph scopes used by Agent tools."""

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from capsule.db.repositories import RelationGraphRepository

router = APIRouter(prefix="/api/v1/graphs", tags=["graphs"])


class NarrativeGraphCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=64)
    name: str = Field(default="未命名图谱", min_length=1, max_length=1024)
    description: str = Field(default="", max_length=20_000)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized


class NarrativeGraphResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_id: str
    workspace_id: str
    name: str
    description: str


def _repository(request: Request) -> RelationGraphRepository:
    repository = getattr(request.app.state, "relation_graph_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "graph_repository_not_ready",
                "message": "Narrative graph repository is not ready",
            },
        )
    return cast(RelationGraphRepository, repository)


@router.post("", response_model=NarrativeGraphResponse, status_code=status.HTTP_201_CREATED)
async def create_narrative_graph(
    payload: NarrativeGraphCreateRequest,
    request: Request,
) -> NarrativeGraphResponse:
    try:
        graph = await _repository(request).create_graph(
            workspace_id=payload.workspace_id,
            name=payload.name,
            description=payload.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return NarrativeGraphResponse.model_validate(graph)
