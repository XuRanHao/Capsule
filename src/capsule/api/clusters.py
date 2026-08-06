"""Asynchronous front-end API for one Embedding Type clustering run."""

import logging
from typing import Protocol, cast
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from capsule.db.repositories import ClusterRepository, NewAssetClusterStatusRecord
from capsule.enums import (
    ClusterMemberSource,
    ClusterMode,
    ClusterRunStatus,
    EmbeddingType,
    NewAssetClusterStatus,
)
from capsule.pipeline.cluster_service import ClusterService
from capsule.schemas import (
    ClusterCapsuleRecord,
    ClusterMemberRecord,
    ClusterRunListResponse,
    ClusterRunRecord,
    CurrentClusterListResponse,
    CurrentClusterMemberListResponse,
    CurrentClusterMemberMutation,
    CurrentClusterMemberMutationResponse,
    CurrentClusterMemberRecord,
    CurrentClusterPatch,
    CurrentClusterRecord,
    NewAssetClusterStatusItem,
    NewAssetClusterStatusResponse,
)

router = APIRouter(prefix="/api/v1", tags=["cluster-runs"])
logger = logging.getLogger(__name__)


class ClusterRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1, max_length=64)
    embedding_type: EmbeddingType = EmbeddingType.NATIVE_MULTIMODAL
    pca_dimension: int = Field(default=8, ge=2, le=1024)
    min_samples: int = Field(default=3, ge=1, le=10_000)
    min_cluster_size: int = Field(default=3, ge=2, le=10_000)
    optimize_parameters: bool = False


class ClusterRunSubmission(BaseModel):
    cluster_run_id: str
    status: ClusterRunStatus = ClusterRunStatus.PENDING


class ClusterCapsuleListResponse(BaseModel):
    items: list[ClusterCapsuleRecord] = Field(default_factory=list)


class ClusterMemberListResponse(BaseModel):
    items: list[ClusterMemberRecord] = Field(default_factory=list)


class ClusterCapsuleOverridePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=1024)
    description: str | None = Field(default=None, max_length=20_000)


class CurrentClusterRepositoryProtocol(Protocol):
    async def get_new_asset_cluster_status(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> NewAssetClusterStatusRecord: ...

    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: list[ClusterMode] | None = None,
    ) -> list[CurrentClusterRecord]: ...

    async def get_cluster(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> CurrentClusterRecord: ...

    async def list_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> list[CurrentClusterMemberRecord]: ...

    async def set_mode(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        mode: ClusterMode,
    ) -> CurrentClusterRecord: ...

    async def set_name(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        name: str,
    ) -> CurrentClusterRecord: ...

    async def attach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: list[str],
        source: ClusterMemberSource = ClusterMemberSource.USER,
        scores: dict[str, float] | None = None,
    ) -> list[CurrentClusterMemberRecord]: ...

    async def detach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: list[str],
        created_by: str | None = None,
    ) -> list[str]: ...


class CurrentClusterProcessorProtocol(Protocol):
    async def process_assets(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        asset_ids: list[str],
    ) -> object: ...


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


def _current_cluster_repository(request: Request) -> CurrentClusterRepositoryProtocol:
    repository = getattr(request.app.state, "current_cluster_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "current_cluster_repository_not_ready",
                "message": "current cluster persistence is not ready",
            },
        )
    return cast(CurrentClusterRepositoryProtocol, repository)


@router.get("/clusters", response_model=CurrentClusterListResponse, tags=["clusters"])
async def list_current_clusters(
    request: Request,
    embedding_type: EmbeddingType,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> CurrentClusterListResponse:
    items = await _current_cluster_repository(request).list_clusters(
        workspace_id=workspace_id,
        embedding_type=embedding_type.value,
    )
    return CurrentClusterListResponse(items=items)


@router.get(
    "/clusters/assets/status",
    response_model=NewAssetClusterStatusResponse,
    tags=["clusters"],
)
async def get_new_asset_cluster_status(
    request: Request,
    embedding_type: EmbeddingType,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> NewAssetClusterStatusResponse:
    settings = request.app.state.settings
    result = await _current_cluster_repository(request).get_new_asset_cluster_status(
        workspace_id=workspace_id,
        embedding_type=embedding_type.value,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        milvus_collection=settings.milvus_collection,
    )
    incrementally_clustered_count = sum(
        item.status is NewAssetClusterStatus.INCREMENTALLY_CLUSTERED
        for item in result.items
    )
    pending_count = sum(
        item.status is NewAssetClusterStatus.PENDING for item in result.items
    )
    manual_management_count = sum(
        item.status is NewAssetClusterStatus.MANUAL_MANAGEMENT for item in result.items
    )
    return NewAssetClusterStatusResponse(
        workspace_id=workspace_id,
        embedding_type=embedding_type,
        initialized=result.has_baseline,
        bootstrap_minimum_count=settings.cluster_bootstrap_minimum_count,
        baseline_cluster_run_id=result.baseline_cluster_run_id,
        baseline_sample_count=result.baseline_sample_count,
        eligible_asset_count=result.eligible_asset_count,
        new_asset_count=len(result.items),
        incrementally_clustered_count=incrementally_clustered_count,
        pending_count=pending_count,
        manual_management_count=manual_management_count,
        items=[
            NewAssetClusterStatusItem.model_validate(item, from_attributes=True)
            for item in result.items
        ],
    )


@router.get(
    "/clusters/{cluster_id}/members",
    response_model=CurrentClusterMemberListResponse,
    tags=["clusters"],
)
async def list_current_cluster_members(
    cluster_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> CurrentClusterMemberListResponse:
    repository = _current_cluster_repository(request)
    try:
        # Verify ownership so an empty member list does not hide an unknown Cluster.
        await repository.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        items = await repository.list_members(
            cluster_id=cluster_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise _current_cluster_not_found(cluster_id) from exc
    return CurrentClusterMemberListResponse(items=items)


@router.patch(
    "/clusters/{cluster_id}",
    response_model=CurrentClusterRecord,
    tags=["clusters"],
)
async def patch_current_cluster(
    cluster_id: str,
    payload: CurrentClusterPatch,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> CurrentClusterRecord:
    repository = _current_cluster_repository(request)
    try:
        updated: CurrentClusterRecord | None = None
        if payload.mode is not None:
            updated = await repository.set_mode(
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                mode=payload.mode,
            )
        if payload.name is not None:
            updated = await repository.set_name(
                cluster_id=cluster_id,
                workspace_id=workspace_id,
                name=payload.name,
            )
        if updated is None:  # pragma: no cover - guarded by request validation
            raise ValueError("current cluster patch is empty")
        return updated
    except ValueError as exc:
        raise _current_cluster_not_found(cluster_id) from exc


@router.post(
    "/clusters/{cluster_id}/members:attach",
    response_model=CurrentClusterMemberMutationResponse,
    tags=["clusters"],
)
async def attach_current_cluster_members(
    cluster_id: str,
    payload: CurrentClusterMemberMutation,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> CurrentClusterMemberMutationResponse:
    try:
        await _current_cluster_repository(request).attach_members(
            cluster_id=cluster_id,
            workspace_id=workspace_id,
            asset_ids=payload.asset_ids,
            source=ClusterMemberSource.USER,
        )
    except ValueError as exc:
        raise _current_cluster_mutation_error(cluster_id, exc) from exc
    return CurrentClusterMemberMutationResponse(
        cluster_id=cluster_id,
        asset_ids=payload.asset_ids,
    )


@router.post(
    "/clusters/{cluster_id}/members:detach",
    response_model=CurrentClusterMemberMutationResponse,
    tags=["clusters"],
)
async def detach_current_cluster_members(
    cluster_id: str,
    payload: CurrentClusterMemberMutation,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> CurrentClusterMemberMutationResponse:
    repository = _current_cluster_repository(request)
    try:
        cluster = await repository.get_cluster(
            cluster_id=cluster_id,
            workspace_id=workspace_id,
        )
        await repository.detach_members(
            cluster_id=cluster_id,
            workspace_id=workspace_id,
            asset_ids=payload.asset_ids,
        )
    except ValueError as exc:
        raise _current_cluster_mutation_error(cluster_id, exc) from exc
    processor = cast(
        CurrentClusterProcessorProtocol | None,
        getattr(request.app.state, "incremental_cluster_coordinator", None),
    )
    if processor is not None:
        try:
            await processor.process_assets(
                workspace_id=workspace_id,
                embedding_type=EmbeddingType(cluster.embedding_type),
                asset_ids=payload.asset_ids,
            )
        except Exception as exc:
            logger.warning(
                "post-detach incremental clustering failed for cluster=%s: %s",
                cluster_id,
                str(exc) or type(exc).__name__,
            )
    return CurrentClusterMemberMutationResponse(
        cluster_id=cluster_id,
        asset_ids=payload.asset_ids,
    )


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
            preprocessing={
                "trigger": "user",
                "requested_pca_dimension": payload.pca_dimension,
                "parameter_selection": (
                    "user_defined_selection_optimized"
                    if payload.optimize_parameters
                    else "user_defined"
                ),
            },
            parameters={
                "min_samples": payload.min_samples,
                "min_cluster_size": payload.min_cluster_size,
            },
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
        pca_dimension=payload.pca_dimension,
        min_samples=payload.min_samples,
        min_cluster_size=payload.min_cluster_size,
        optimize_parameters=payload.optimize_parameters,
    )
    return ClusterRunSubmission(cluster_run_id=cluster_run_id)


@router.get("/cluster-runs", response_model=ClusterRunListResponse)
async def list_cluster_runs(
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
) -> ClusterRunListResponse:
    return ClusterRunListResponse(
        items=await _cluster_repository(request).list_runs(
            workspace_id=workspace_id,
            limit=limit,
        )
    )


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
    "/cluster-capsules/{cluster_capsule_id}/members",
    response_model=ClusterMemberListResponse,
)
async def list_cluster_members(
    cluster_capsule_id: str,
    request: Request,
    workspace_id: str = Query(min_length=1, max_length=64),
) -> ClusterMemberListResponse:
    try:
        members = await _cluster_repository(request).list_members(
            cluster_capsule_id=cluster_capsule_id,
            workspace_id=workspace_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "cluster_capsule_not_found",
                "message": str(exc),
            },
        ) from exc
    query_string = urlencode({"workspace_id": workspace_id})
    return ClusterMemberListResponse(
        items=[
            member.model_copy(
                update={
                    "preview_url": (
                        f"{request.url_for('get_asset_thumbnail', asset_id=member.asset_id).path}"
                        f"?{query_string}"
                    )
                    if member.asset_type.value in {"image", "video_segment"}
                    else None
                }
            )
            for member in members
        ]
    )


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
    pca_dimension: int,
    min_samples: int,
    min_cluster_size: int,
    optimize_parameters: bool,
) -> None:
    await service.run(
        workspace_id=workspace_id,
        embedding_type=embedding_type,
        cluster_run_id=cluster_run_id,
        pca_dimension=pca_dimension,
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        optimize_parameters=optimize_parameters,
    )


def _run_not_found(cluster_run_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "cluster_run_not_found",
            "message": f"Cluster run {cluster_run_id!r} was not found",
        },
    )


def _current_cluster_not_found(cluster_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "code": "current_cluster_not_found",
            "message": f"Current cluster {cluster_id!r} was not found",
        },
    )


def _current_cluster_mutation_error(cluster_id: str, exc: ValueError) -> HTTPException:
    message = str(exc)
    normalized = message.lower()
    if "cluster" in normalized and any(
        marker in normalized
        for marker in ("not found", "does not exist", "another workspace")
    ):
        return _current_cluster_not_found(cluster_id)
    if any(
        marker in normalized
        for marker in ("resident", "conflict", "already assigned", "another cluster", "locked")
    ):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "cluster_membership_conflict", "message": message},
        )
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={"code": "invalid_cluster_membership_change", "message": message},
    )
