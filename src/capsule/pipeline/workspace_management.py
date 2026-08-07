"""Workspace lifecycle operations scoped to exactly one workspace."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

from capsule.config import Settings
from capsule.db.repositories import WorkspaceDeleteSnapshot
from capsule.schemas import WorkspaceDeleteResult, WorkspaceRecord


class WorkspaceRepository(Protocol):
    async def list_workspaces(self) -> list[WorkspaceRecord]: ...
    async def create_workspace(
        self, *, name: str, workspace_id: str | None = None
    ) -> WorkspaceRecord: ...
    async def delete_workspace_records(
        self, *, workspace_id: str
    ) -> WorkspaceDeleteSnapshot: ...


class WorkspaceVectorStore(Protocol):
    async def delete_workspace(self, workspace_id: str) -> int: ...


class WorkspaceObjectStorage(Protocol):
    async def delete_uris(self, uris: list[str]) -> int: ...
    async def delete_object_keys(self, object_keys: list[str]) -> int: ...


class WorkspaceService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: WorkspaceRepository,
        vector_store: WorkspaceVectorStore,
        object_storage: WorkspaceObjectStorage,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._vector_store = vector_store
        self._object_storage = object_storage

    async def list(self) -> list[WorkspaceRecord]:
        return await self._repository.list_workspaces()

    async def create(self, *, name: str, workspace_id: str | None) -> WorkspaceRecord:
        return await self._repository.create_workspace(name=name, workspace_id=workspace_id)

    async def delete(self, *, workspace_id: str, cancelled_jobs: int = 0) -> WorkspaceDeleteResult:
        snapshot = await self._repository.delete_workspace_records(workspace_id=workspace_id)
        warnings: list[str] = []
        s3_uris = [uri for uri in snapshot.storage_uris if uri.startswith("s3://")]
        local_uris = [uri for uri in snapshot.storage_uris if uri.startswith("file://")]

        vector_count = await _best_effort(
            lambda: self._vector_store.delete_workspace(workspace_id), "Milvus 向量", warnings
        )
        object_count = await _best_effort(
            lambda: self._object_storage.delete_uris(s3_uris), "MinIO 素材文件", warnings
        )
        object_count += await _best_effort(
            lambda: self._object_storage.delete_object_keys(list(snapshot.object_keys)),
            "MinIO 查询图片",
            warnings,
        )
        staging_count = await _best_effort(
            lambda: asyncio.to_thread(
                _delete_workspace_paths,
                self._settings,
                list(snapshot.staging_paths),
                local_uris,
            ),
            "本地工作空间文件",
            warnings,
        )
        return WorkspaceDeleteResult(
            workspace_id=workspace_id,
            workspace_deleted=True,
            assets_deleted=snapshot.asset_count,
            source_files_deleted=snapshot.source_file_count,
            embeddings_deleted=snapshot.embedding_count,
            jobs_deleted=snapshot.job_count,
            vectors_deleted=vector_count,
            objects_deleted=object_count,
            staging_paths_deleted=staging_count,
            cancelled_jobs=cancelled_jobs,
            cleanup_warnings=warnings,
        )


async def _best_effort(
    operation: Callable[[], Awaitable[int]], label: str, warnings: list[str]
) -> int:
    try:
        return int(await operation())
    except Exception as exc:
        warnings.append(f"{label} 清理失败：{type(exc).__name__}")
        return 0


def _delete_workspace_paths(
    settings: Settings, staging_paths: list[str], local_uris: list[str]
) -> int:
    managed_roots = (
        settings.import_root.expanduser().resolve(),
        settings.document_media_root.expanduser().resolve(),
    )
    candidates = [Path(path).expanduser().resolve() for path in staging_paths]
    candidates.extend(
        Path(unquote(urlparse(uri).path)).resolve() for uri in local_uris
    )
    deleted = 0
    for path in sorted(set(candidates), key=lambda item: len(item.parts), reverse=True):
        if not any(path.is_relative_to(root) and path != root for root in managed_roots):
            continue
        if path.is_dir():
            shutil.rmtree(path)
            deleted += 1
        elif path.exists():
            path.unlink()
            deleted += 1
    return deleted
