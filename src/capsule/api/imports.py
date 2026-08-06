"""Three-stage browser folder import API."""

from typing import Annotated, cast

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from capsule.db.repositories import AssetRepository
from capsule.pipeline.import_service import (
    BrowserImportService,
    ImportCompletion,
    ImportFileTooLargeError,
    ImportSubmissionError,
)
from capsule.schemas import ProcessingJobListResponse, ProcessingJobRecord

router = APIRouter(prefix="/api/v1", tags=["imports"])


class CreateImportJobRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=64)


class ImportJobCreated(BaseModel):
    job_id: str
    status: str


class ImportFileUploaded(BaseModel):
    job_id: str
    relative_path: str
    size_bytes: int


class ImportJobStarted(BaseModel):
    job_id: str
    status: str
    file_count: int


class ImportJobsCleared(BaseModel):
    deleted_count: int
    cancelled_count: int


def _import_service(request: Request) -> BrowserImportService:
    service = getattr(request.app.state, "import_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "import_service_not_ready", "message": "import service is not ready"},
        )
    return cast(BrowserImportService, service)


def _asset_repository(request: Request) -> AssetRepository:
    repository = getattr(request.app.state, "asset_repository", None)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "import_repository_not_ready",
                "message": "import storage is not ready",
            },
        )
    return cast(AssetRepository, repository)


@router.post(
    "/import-jobs",
    response_model=ImportJobCreated,
    status_code=status.HTTP_201_CREATED,
)
async def create_import_job(
    payload: CreateImportJobRequest,
    request: Request,
) -> ImportJobCreated:
    """Create an empty job; no browser bytes are transferred in this request."""
    try:
        job = await _import_service(request).create_job(workspace_id=payload.workspace_id)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "import_staging_unavailable", "message": str(exc)},
        ) from exc
    return ImportJobCreated(job_id=job.job_id, status="queued")


@router.get("/import-jobs", response_model=ProcessingJobListResponse)
async def list_import_jobs(
    request: Request,
    workspace_id: str,
    limit: int = 50,
) -> ProcessingJobListResponse:
    items = await _asset_repository(request).list_jobs(
        workspace_id=workspace_id,
        limit=max(1, min(limit, 200)),
    )
    return ProcessingJobListResponse(items=items)


@router.delete("/import-jobs", response_model=ImportJobsCleared)
async def clear_import_jobs(
    request: Request,
    workspace_id: str,
) -> ImportJobsCleared:
    cancelled_count = await _import_service(request).cancel_active_jobs(
        workspace_id=workspace_id
    )
    deleted_count = await _asset_repository(request).clear_jobs(workspace_id=workspace_id)
    return ImportJobsCleared(
        deleted_count=deleted_count,
        cancelled_count=cancelled_count,
    )


@router.post(
    "/import-jobs/{job_id}/files",
    response_model=ImportFileUploaded,
    status_code=status.HTTP_201_CREATED,
)
async def upload_import_file(
    job_id: str,
    request: Request,
    workspace_id: Annotated[str, Form(min_length=1, max_length=64)],
    relative_path: Annotated[str, Form(min_length=1)],
    file: Annotated[UploadFile, File()],
) -> ImportFileUploaded:
    """Upload one source file. Repeating the same relative path replaces it safely."""
    try:
        size_bytes = await _import_service(request).upload_file(
            job_id=job_id,
            workspace_id=workspace_id,
            file=file,
            relative_path=relative_path,
        )
    except ImportFileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "import_file_too_large", "message": str(exc)},
        ) from exc
    except ImportSubmissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_import_file", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_job_not_found", "message": str(exc)},
        ) from exc
    return ImportFileUploaded(job_id=job_id, relative_path=relative_path, size_bytes=size_bytes)


@router.post(
    "/import-jobs/{job_id}/complete",
    response_model=ImportJobStarted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def complete_import_job(
    job_id: str,
    payload: CreateImportJobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ImportJobStarted:
    """Start assetization only after the browser has confirmed all uploads."""
    service = _import_service(request)
    try:
        completion = await service.complete_job(
            job_id=job_id,
            workspace_id=payload.workspace_id,
        )
    except ImportSubmissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_import_completion", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_job_not_found", "message": str(exc)},
        ) from exc
    background_tasks.add_task(
        _execute_import,
        service=service,
        completion=completion,
        workspace_id=payload.workspace_id,
    )
    return ImportJobStarted(
        job_id=completion.job_id,
        status="running",
        file_count=completion.file_count,
    )


@router.get("/import-jobs/{job_id}", response_model=ProcessingJobRecord)
async def get_import_job(
    job_id: str,
    request: Request,
    workspace_id: str,
) -> ProcessingJobRecord:
    try:
        return await _asset_repository(request).get_job(job_id=job_id, workspace_id=workspace_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_job_not_found", "message": str(exc)},
        ) from exc


async def _execute_import(
    *,
    service: BrowserImportService,
    completion: ImportCompletion,
    workspace_id: str,
) -> None:
    await service.execute(completion=completion, workspace_id=workspace_id)
