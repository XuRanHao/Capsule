"""Destructive full-library cleanup used by the Asset library."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from capsule.config import Settings
from capsule.db.repositories import LibraryClearSnapshot
from capsule.schemas import LibraryClearResult


class LibraryClearRepository(Protocol):
    async def clear_all_records(self) -> LibraryClearSnapshot: ...


class LibraryVectorStore(Protocol):
    async def delete_all(self) -> int: ...


class LibraryObjectStorage(Protocol):
    async def delete_all_objects(self) -> int: ...


#===========================================
#      Full library reset across persistence layers
#===========================================


class LibraryClearService:
    """Clear all persisted Asset-library data across every workspace."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: LibraryClearRepository,
        vector_store: LibraryVectorStore,
        object_storage: LibraryObjectStorage,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vector_store = vector_store
        self._object_storage = object_storage

    async def clear_all(self) -> LibraryClearResult:
        snapshot = await self._repository.clear_all_records()
        warnings: list[str] = []

        vector_count = await _best_effort(
            operation=self._vector_store.delete_all,
            label="Milvus 向量",
            warnings=warnings,
        )
        object_count = await _best_effort(
            operation=self._object_storage.delete_all_objects,
            label="MinIO 文件",
            warnings=warnings,
        )
        local_path_count = await _best_effort(
            operation=lambda: asyncio.to_thread(
                _clear_import_root,
                self._settings.import_root,
            ),
            label="本地导入暂存文件",
            warnings=warnings,
        )

        return LibraryClearResult(
            workspaces_deleted=snapshot.workspace_count,
            assets_deleted=snapshot.asset_count,
            source_files_deleted=snapshot.source_file_count,
            embeddings_deleted=snapshot.embedding_count,
            jobs_deleted=snapshot.job_count,
            vectors_deleted=vector_count,
            objects_deleted=object_count,
            staging_paths_deleted=local_path_count,
            cleanup_warnings=warnings,
        )


async def _best_effort(
    *,
    operation: Callable[[], Awaitable[int]],
    label: str,
    warnings: list[str],
) -> int:
    try:
        return int(await operation())
    except Exception as exc:  # External cleanup must not restore deleted database records.
        warnings.append(f"{label} 清理失败：{type(exc).__name__}")
        return 0


def _clear_import_root(import_root: Path) -> int:
    root = import_root.expanduser().resolve()
    deleted = 0
    if not root.exists():
        return 0
    for path in root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted += 1
    return deleted
