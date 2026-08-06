from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.pipeline.import_service import BrowserImportJob, ImportCompletion
from capsule.schemas import ProcessingJobRecord


class FakeImportService:
    def __init__(self) -> None:
        self.created_workspaces: list[str] = []
        self.uploads: list[dict[str, object]] = []
        self.completions: list[dict[str, object]] = []
        self.executions: list[dict[str, object]] = []
        self.cancelled_workspaces: list[str] = []

    async def create_job(self, *, workspace_id: str) -> BrowserImportJob:
        self.created_workspaces.append(workspace_id)
        return BrowserImportJob(job_id="job_api_import", staged_path=Path("/tmp/import-api-test"))

    async def upload_file(self, **values: object) -> int:
        self.uploads.append(values)
        return 16

    async def complete_job(self, **values: object) -> ImportCompletion:
        self.completions.append(values)
        return ImportCompletion(
            job_id="job_api_import",
            staged_path=Path("/tmp/import-api-test"),
            file_count=1,
        )

    async def execute(self, **values: object) -> None:
        self.executions.append(values)

    async def cancel_active_jobs(self, *, workspace_id: str) -> int:
        self.cancelled_workspaces.append(workspace_id)
        return 1


class FakeAssetRepository:
    def __init__(self) -> None:
        self.cleared_workspaces: list[str] = []

    async def clear_jobs(self, *, workspace_id: str) -> int:
        self.cleared_workspaces.append(workspace_id)
        return 4

    async def get_job(self, *, job_id: str, workspace_id: str) -> ProcessingJobRecord:
        assert job_id == "job_api_import"
        assert workspace_id == "workspace_import_api"
        return ProcessingJobRecord(
            job_id=job_id,
            workspace_id=workspace_id,
            input_path="/tmp/import-api-test",
            total_count=1,
            completed_count=0,
            failed_count=0,
            status="running",
            current_stage="parsing",
            error_info=[],
            started_at=None,
            completed_at=None,
        )


@pytest.mark.asyncio
async def test_import_api_creates_then_uploads_then_starts_one_job() -> None:
    import_service = FakeImportService()
    app = create_app(
        settings=Settings(),
        import_service=import_service,  # type: ignore[arg-type]
        asset_repository=FakeAssetRepository(),  # type: ignore[arg-type]
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/import-jobs",
                json={"workspace_id": "workspace_import_api"},
            )
            uploaded = await client.post(
                "/api/v1/import-jobs/job_api_import/files",
                data={
                    "workspace_id": "workspace_import_api",
                    "relative_path": "folder/note.md",
                },
                files={"file": ("note.md", b"# Imported note", "text/markdown")},
            )
            started = await client.post(
                "/api/v1/import-jobs/job_api_import/complete",
                json={"workspace_id": "workspace_import_api"},
            )
            job = await client.get(
                "/api/v1/import-jobs/job_api_import",
                params={"workspace_id": "workspace_import_api"},
            )

    assert created.status_code == 201
    assert created.json() == {"job_id": "job_api_import", "status": "queued"}
    assert uploaded.status_code == 201
    assert uploaded.json() == {
        "job_id": "job_api_import",
        "relative_path": "folder/note.md",
        "size_bytes": 16,
    }
    assert started.status_code == 202
    assert started.json() == {"job_id": "job_api_import", "status": "running", "file_count": 1}
    assert import_service.created_workspaces == ["workspace_import_api"]
    assert import_service.uploads[0]["relative_path"] == "folder/note.md"
    assert import_service.completions[0]["workspace_id"] == "workspace_import_api"
    assert import_service.executions[0]["workspace_id"] == "workspace_import_api"
    assert job.status_code == 200
    assert job.json()["status"] == "running"


@pytest.mark.asyncio
async def test_import_api_cancels_active_jobs_then_clears_every_job() -> None:
    repository = FakeAssetRepository()
    import_service = FakeImportService()
    app = create_app(
        settings=Settings(),
        import_service=import_service,  # type: ignore[arg-type]
        asset_repository=repository,  # type: ignore[arg-type]
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete(
                "/api/v1/import-jobs",
                params={"workspace_id": "workspace_import_api"},
            )

    assert response.status_code == 200
    assert response.json() == {"deleted_count": 4, "cancelled_count": 1}
    assert repository.cleared_workspaces == ["workspace_import_api"]
    assert import_service.cancelled_workspaces == ["workspace_import_api"]
