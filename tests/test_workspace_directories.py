from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.db.workspace_directories import WorkspaceDirectoryWorkspaceNotFoundError
from capsule.pipeline.workspace_directories import (
    WorkspaceDirectoryService,
    normalize_workspace_directory_path,
)
from capsule.schemas import WorkspaceDirectoryRecord


class FakeDirectoryRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.records: dict[str, WorkspaceDirectoryRecord] = {}

    async def list_directories(self, *, workspace_id: str) -> list[WorkspaceDirectoryRecord]:
        if workspace_id == "missing":
            raise WorkspaceDirectoryWorkspaceNotFoundError("workspace does not exist: missing")
        return list(self.records.values())

    async def ensure_directories(
        self,
        *,
        workspace_id: str,
        paths: tuple[str, ...],
    ) -> list[WorkspaceDirectoryRecord]:
        if workspace_id == "missing":
            raise WorkspaceDirectoryWorkspaceNotFoundError("workspace does not exist: missing")
        self.calls.append((workspace_id, paths))
        now = datetime.now(UTC)
        for path in paths:
            self.records.setdefault(
                path,
                WorkspaceDirectoryRecord(path=path, created_at=now, updated_at=now),
            )
        return [self.records[path] for path in paths]


@pytest.mark.asyncio
async def test_directory_service_normalizes_path_and_creates_missing_parents() -> None:
    repository = FakeDirectoryRepository()
    service = WorkspaceDirectoryService(repository=repository)

    result = await service.create(workspace_id="workspace_1", path=r"角色//NPC/./立绘/")

    assert repository.calls == [("workspace_1", ("角色", "角色/NPC", "角色/NPC/立绘"))]
    assert [directory.path for directory in result] == ["角色", "角色/NPC", "角色/NPC/立绘"]


def test_directory_path_rejects_escape_and_invalid_paths() -> None:
    assert normalize_workspace_directory_path(r"角色\NPC//立绘/") == "角色/NPC/立绘"
    for path in ("", " /root", "角色/../秘密", r"C:\root", "角色/\x00bad"):
        try:
            normalize_workspace_directory_path(path)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid path: {path!r}")


def test_directory_api_returns_parent_paths_and_normalizes_errors() -> None:
    repository = FakeDirectoryRepository()
    app = create_app(
        settings=Settings(),
        workspace_directory_service=WorkspaceDirectoryService(repository=repository),
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/workspaces/workspace_1/directories",
            json={"path": r"资料\角色/立绘"},
        )
        listed = client.get("/api/v1/workspaces/workspace_1/directories")
        invalid = client.post(
            "/api/v1/workspaces/workspace_1/directories",
            json={"path": "资料/../秘密"},
        )
        missing = client.get("/api/v1/workspaces/missing/directories")

    assert created.status_code == 201
    assert [item["path"] for item in created.json()["items"]] == [
        "资料",
        "资料/角色",
        "资料/角色/立绘",
    ]
    assert listed.status_code == 200
    assert [item["path"] for item in listed.json()["items"]] == [
        "资料",
        "资料/角色",
        "资料/角色/立绘",
    ]
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_workspace_directory_path"
    assert missing.status_code == 404
