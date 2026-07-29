"""Asynchronous front-end API for one Embedding Type clustering run."""

from typing import cast

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from capsule.db.repositories import ClusterRepository
from capsule.enums import ClusterRunStatus, EmbeddingType
from capsule.pipeline.cluster_service import ClusterService
from capsule.schemas import ClusterCapsuleRecord, ClusterRunRecord

router = APIRouter(prefix="/api/v1", tags=["cluster-runs"])


class ClusterRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=64)
    embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL


class ClusterRunSubmission(BaseModel):
    cluster_run_id: str
    status: ClusterRunStatus = ClusterRunStatus.PENDING


class ClusterCapsuleListResponse(BaseModel):
    items: list[ClusterCapsuleRecord] = Field(default_factory=list)


class ClusterCapsuleOverridePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=1024)
    description: str | None = Field(default=None, max_length=20_000)


def _cluster_service(request: Request) -> ClusterService:
    service = getattr(request.app.state, "cluster_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "cluster_service_not_ready",
                "message": "cluster service requires CAPSULE_ARK_API_KEY",
            },
        )
    return cast(ClusterService, service)


def _cluster_repository(request: Request) -> ClusterRepository:
    repository = getattr(request.app.state, "cluster_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "cluster_repository_not_ready",
                "message": "cluster persistence is not ready",
            },
        )
    return cast(ClusterRepository, repository)


@router.post(
    "/cluster-runs",
    response_model=ClusterRunSubmission,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_cluster_run(
    payload: ClusterRunCreate,
    background_tasks: BackgroundTasks,
    request: Request,
) -> ClusterRunSubmission:
    """Queue one Type and return its durable run ID without waiting for model calls."""
    service = _cluster_service(request)
    repository = _cluster_repository(request)
    try:
        cluster_run_id = await repository.create_pending_run(
            workspace_id=payload.workspace_id,
            embedding_type=payload.embedding_type.value,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "workspace_not_found", "message": str(exc)},
        ) from exc
    background_tasks.add_task(
        _execute_cluster_run,
        service=service,
        workspace_id=payload.workspace_id,
        embedding_type=payload.embedding_type,
        cluster_run_id=cluster_run_id,
    )
    return ClusterRunSubmission(cluster_run_id=cluster_run_id)


@router.get("/cluster-runs/{cluster_run_id}", response_model=ClusterRunRecord)
async def get_cluster_run(
    cluster_run_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> ClusterRunRecord:
    try:
        return await _cluster_repository(request).get_run(
            cluster_run_id=cluster_run_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise _run_not_found(cluster_run_id) from exc


@router.get(
    "/cluster-runs/{cluster_run_id}/capsules",
    response_model=ClusterCapsuleListResponse,
)
async def list_cluster_capsules(
    cluster_run_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> ClusterCapsuleListResponse:
    try:
        # Verify run ownership even before its first Capsule is written.
        await _cluster_repository(request).get_run(
            cluster_run_id=cluster_run_id,
            workspace_id=workspace_id,
        )
        items = await _cluster_repository(request).list_capsules(
            cluster_run_id=cluster_run_id,
            workspace_id=workspace_id,
        )
        return ClusterCapsuleListResponse(items=items)
    except ValueError as exc:
        raise _run_not_found(cluster_run_id) from exc


@router.patch("/cluster-capsules/{cluster_capsule_id}", response_model=ClusterCapsuleRecord)
async def patch_cluster_capsule(
    cluster_capsule_id: str,
    payload: ClusterCapsuleOverridePatch,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> ClusterCapsuleRecord:
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "empty_cluster_capsule_patch",
                "message": "provide name and/or description; null clears an override",
            },
        )
    try:
        return await _cluster_repository(request).update_overrides(
            cluster_capsule_id=cluster_capsule_id,
            workspace_id=workspace_id,
            update_name="name" in fields,
            name=payload.name,
            update_description="description" in fields,
            description=payload.description,
        )
    except ValueError as exc:
        if str(exc).startswith("user override"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "invalid_cluster_capsule_override", "message": str(exc)},
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "cluster_capsule_not_found", "message": str(exc)},
        ) from exc


async def _execute_cluster_run(
    *,
    service: ClusterService,
    workspace_id: str,
    embedding_type: EmbeddingType,
    cluster_run_id: str,
) -> None:
    await service.run(
        workspace_id=workspace_id,
        embedding_type=embedding_type,
        cluster_run_id=cluster_run_id,
    )


def _run_not_found(cluster_run_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "cluster_run_not_found",
            "message": f"Cluster run {cluster_run_id!r} was not found",
        },
    )
