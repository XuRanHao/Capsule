from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from capsule.config import Settings
from capsule.pipeline.import_service import BrowserImportService
from capsule.schemas import ProcessingJobRecord


class FakeAssetRepository:
    def __init__(self) -> None:
        self.import_root: Path | None = None
        self.status = "queued"
        self.started_count: int | None = None

    async def create_pending_import_job(self, **values: object) -> str:
        self.import_root = values["import_root"]  # type: ignore[assignment]
        return "job_import_test"

    async def get_job(self, *, job_id: str, workspace_id: str) -> ProcessingJobRecord:
        assert job_id == "job_import_test"
        assert workspace_id == "workspace_import_test"
        assert self.import_root is not None
        return ProcessingJobRecord(
            job_id=job_id,
            workspace_id=workspace_id,
            input_path=str(self.import_root / job_id),
            total_count=self.started_count or 0,
            completed_count=0,
            failed_count=0,
            status=self.status,
            current_stage="discovering",
            error_info=[],
            started_at=None,
            completed_at=None,
        )

    async def start_import_job(self, *, job_id: str, total_count: int) -> None:
        assert job_id == "job_import_test"
        self.status = "running"
        self.started_count = total_count

    async def fail_job(self, **_: object) -> None:
        raise AssertionError("import should not fail")


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(self, input_path: Path, workspace_id: str, *, job_id: str) -> SimpleNamespace:
        self.calls.append(
            {"input_path": input_path, "workspace_id": workspace_id, "job_id": job_id}
        )
        return SimpleNamespace(job_id=job_id)


@pytest.mark.asyncio
async def test_browser_import_uploads_each_file_before_assetization(tmp_path: Path) -> None:
    repository = FakeAssetRepository()
    runner = FakeRunner()
    service = BrowserImportService(
        settings=Settings(import_root=tmp_path / "imports"),
        repository=repository,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
    )

    job = await service.create_job(workspace_id="workspace_import_test")
    await service.upload_file(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
        file=UploadFile(filename="note.md", file=BytesIO(b"# First version")),
        relative_path="references/notes/note.md",
    )
    await service.upload_file(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
        file=UploadFile(filename="note.md", file=BytesIO(b"# Retried version")),
        relative_path="references/notes/note.md",
    )
    completion = await service.complete_job(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
    )
    await service.execute(completion=completion, workspace_id="workspace_import_test")

    assert (job.staged_path / "references/notes/note.md").read_bytes() == b"# Retried version"
    assert repository.started_count == 1
    assert runner.calls == [
        {
            "input_path": job.staged_path,
            "workspace_id": "workspace_import_test",
            "job_id": "job_import_test",
        }
    ]
