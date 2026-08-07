from datetime import UTC, datetime

from fastapi.testclient import TestClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.schemas import WorkspaceDeleteResult, WorkspaceRecord


class FakeWorkspaceService:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.records = [
            WorkspaceRecord(
                workspace_id="workspace_existing",
                name="已有工作空间",
                created_at=now,
                updated_at=now,
            )
        ]
        self.deleted: list[tuple[str, int]] = []

    async def list(self) -> list[WorkspaceRecord]:
        return self.records

    async def create(self, *, name: str, workspace_id: str | None) -> WorkspaceRecord:
        now = datetime.now(UTC)
        record = WorkspaceRecord(
            workspace_id=workspace_id or "workspace_generated",
            name=name,
            created_at=now,
            updated_at=now,
        )
        self.records.append(record)
        return record

    async def delete(self, *, workspace_id: str, cancelled_jobs: int) -> WorkspaceDeleteResult:
        self.deleted.append((workspace_id, cancelled_jobs))
        return WorkspaceDeleteResult(
            workspace_id=workspace_id,
            workspace_deleted=True,
            assets_deleted=3,
            source_files_deleted=2,
            embeddings_deleted=6,
            jobs_deleted=1,
            vectors_deleted=6,
            objects_deleted=4,
            staging_paths_deleted=1,
            cancelled_jobs=cancelled_jobs,
        )


def test_workspace_list_create_and_scoped_delete() -> None:
    service = FakeWorkspaceService()
    app = create_app(
        settings=Settings(),
        workspace_service=service,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        listed = client.get("/api/v1/workspaces")
        created = client.post(
            "/api/v1/workspaces",
            json={"name": "  新工作空间  ", "workspace_id": "workspace_new"},
        )
        rejected = client.delete(
            "/api/v1/workspaces/workspace_new",
            params={"confirmation": "wrong"},
        )
        deleted = client.delete(
            "/api/v1/workspaces/workspace_new",
            params={"confirmation": "workspace_new"},
        )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["workspace_id"] == "workspace_existing"
    assert created.status_code == 201
    assert created.json()["name"] == "新工作空间"
    assert rejected.status_code == 422
    assert service.deleted == [("workspace_new", 0)]
    assert deleted.status_code == 200
    assert deleted.json()["workspace_id"] == "workspace_new"
    assert deleted.json()["assets_deleted"] == 3


def test_workspace_create_rejects_unsafe_id_and_blank_name() -> None:
    app = create_app(
        settings=Settings(),
        workspace_service=FakeWorkspaceService(),  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        unsafe_id = client.post(
            "/api/v1/workspaces", json={"name": "test", "workspace_id": "../all"}
        )
        blank_name = client.post("/api/v1/workspaces", json={"name": "   "})
    assert unsafe_id.status_code == 422
    assert blank_name.status_code == 422
