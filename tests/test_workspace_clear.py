from pathlib import Path

import pytest

from capsule.config import Settings
from capsule.db.repositories import LibraryClearSnapshot
from capsule.pipeline.workspace_clear import LibraryClearService


class FakeRepository:
    def __init__(self, snapshot: LibraryClearSnapshot) -> None:
        self.snapshot = snapshot

    async def clear_all_records(self) -> LibraryClearSnapshot:
        return self.snapshot


class FakeVectorStore:
    def __init__(self) -> None:
        self.clear_calls = 0

    async def delete_all(self) -> int:
        self.clear_calls += 1
        return 6


class FakeObjectStorage:
    def __init__(self) -> None:
        self.clear_calls = 0

    async def delete_all_objects(self) -> int:
        self.clear_calls += 1
        return 2


@pytest.mark.asyncio
async def test_library_clear_removes_external_records_and_staging(tmp_path: Path) -> None:
    import_root = tmp_path / "imports"
    staged_path = import_root / "job_01"
    staged_path.mkdir(parents=True)
    (staged_path / "source.md").write_text("# source", encoding="utf-8")
    snapshot = LibraryClearSnapshot(
        workspace_count=2,
        asset_count=3,
        source_file_count=1,
        embedding_count=6,
        job_count=1,
    )
    repository = FakeRepository(snapshot)
    vectors = FakeVectorStore()
    storage = FakeObjectStorage()
    service = LibraryClearService(
        settings=Settings(import_root=import_root),
        repository=repository,
        vector_store=vectors,
        object_storage=storage,
    )

    result = await service.clear_all()

    assert result.workspaces_deleted == 2
    assert result.assets_deleted == 3
    assert result.vectors_deleted == 6
    assert result.objects_deleted == 2
    assert result.staging_paths_deleted == 1
    assert result.cleanup_warnings == []
    assert vectors.clear_calls == 1
    assert storage.clear_calls == 1
    assert not staged_path.exists()
