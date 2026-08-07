from pathlib import Path

import pytest

from capsule.config import Settings
from capsule.db.repositories import WorkspaceDeleteSnapshot
from capsule.pipeline.workspace_management import WorkspaceService


class FakeRepository:
    def __init__(self, snapshot: WorkspaceDeleteSnapshot) -> None:
        self.snapshot = snapshot
        self.deleted: list[str] = []

    async def delete_workspace_records(self, *, workspace_id: str) -> WorkspaceDeleteSnapshot:
        self.deleted.append(workspace_id)
        return self.snapshot


class FakeVectors:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_workspace(self, workspace_id: str) -> int:
        self.deleted.append(workspace_id)
        return 7


class FakeStorage:
    def __init__(self) -> None:
        self.uris: list[str] = []
        self.keys: list[str] = []

    async def delete_uris(self, uris: list[str]) -> int:
        self.uris.extend(uris)
        return len(uris)

    async def delete_object_keys(self, object_keys: list[str]) -> int:
        self.keys.extend(object_keys)
        return len(object_keys)


@pytest.mark.asyncio
async def test_delete_workspace_cleans_only_snapshot_targets(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    selected_job = import_root / "job_selected"
    other_job = import_root / "job_other"
    selected_job.mkdir(parents=True)
    other_job.mkdir(parents=True)
    (selected_job / "source.png").write_bytes(b"selected")
    (other_job / "source.png").write_bytes(b"other")
    media_root = tmp_path / "document-media"
    selected_media = media_root / "selected.png"
    selected_media.parent.mkdir(parents=True)
    selected_media.write_bytes(b"media")
    snapshot = WorkspaceDeleteSnapshot(
        workspace_id="workspace_selected",
        asset_count=3,
        source_file_count=2,
        embedding_count=7,
        job_count=1,
        storage_uris=(
            "s3://capsule/assets/selected/preview.jpg",
            selected_media.as_uri(),
        ),
        object_keys=("query-images/workspace_selected/query.png",),
        staging_paths=(str(selected_job),),
    )
    repository = FakeRepository(snapshot)
    vectors = FakeVectors()
    storage = FakeStorage()
    service = WorkspaceService(
        settings=Settings(import_root=import_root, document_media_root=media_root),
        repository=repository,  # type: ignore[arg-type]
        vector_store=vectors,
        object_storage=storage,
    )

    result = await service.delete(workspace_id="workspace_selected", cancelled_jobs=1)

    assert repository.deleted == ["workspace_selected"]
    assert vectors.deleted == ["workspace_selected"]
    assert storage.uris == ["s3://capsule/assets/selected/preview.jpg"]
    assert storage.keys == ["query-images/workspace_selected/query.png"]
    assert not selected_job.exists()
    assert not selected_media.exists()
    assert other_job.exists()
    assert result.workspace_id == "workspace_selected"
    assert result.cancelled_jobs == 1
    assert result.objects_deleted == 2
    assert result.staging_paths_deleted == 2
