import re
from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError

from capsule.pipeline.import_service import BrowserImportService
from capsule.pipeline.workspace_management import WorkspaceService
from capsule.schemas import WorkspaceDeleteResult, WorkspaceListResponse, WorkspaceRecord

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspaces"])
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class WorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    workspace_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("workspace name must not be blank")
        return normalized

    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, value: str | None) -> str | None:
        if value is not None and _WORKSPACE_ID.fullmatch(value) is None:
            raise ValueError(
                "workspace_id may contain only letters, digits, underscores and dashes"
            )
        return value


def _service(request: Request) -> WorkspaceService:
    service = getattr(request.app.state, "workspace_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "workspace_service_not_ready",
                "message": "workspace service is not ready",
            },
        )
    return cast(WorkspaceService, service)


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces(request: Request) -> WorkspaceListResponse:
    return WorkspaceListResponse(items=await _service(request).list())


@router.post("", response_model=WorkspaceRecord, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreateRequest, request: Request
) -> WorkspaceRecord:
    try:
        return await _service(request).create(
            name=payload.name,
            workspace_id=payload.workspace_id,
        )
    except IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "workspace_already_exists", "message": "workspace_id already exists"},
        ) from exc


@router.delete("/{workspace_id}", response_model=WorkspaceDeleteResult)
async def delete_workspace(
    workspace_id: str,
    request: Request,
    confirmation: str = Query(min_length=1, max_length=64),
) -> WorkspaceDeleteResult:
    if confirmation != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "workspace_delete_confirmation_required",
                "message": "confirmation must exactly match workspace_id",
            },
        )
    import_service = getattr(request.app.state, "import_service", None)
    cancelled_jobs = 0
    if import_service is not None:
        cancelled_jobs = await cast(BrowserImportService, import_service).cancel_active_jobs(
            workspace_id=workspace_id
        )
    try:
        return await _service(request).delete(
            workspace_id=workspace_id,
            cancelled_jobs=cancelled_jobs,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "workspace_not_found", "message": str(exc)},
        ) from exc
