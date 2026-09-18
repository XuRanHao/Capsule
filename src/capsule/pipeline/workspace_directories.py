"""Business rules for the workspace's user-created directory skeleton."""

from __future__ import annotations

import re
from typing import Protocol

from capsule.schemas import WorkspaceDirectoryRecord


class WorkspaceDirectoryRepository(Protocol):
    async def list_directories(self, *, workspace_id: str) -> list[WorkspaceDirectoryRecord]: ...

    async def ensure_directories(
        self,
        *,
        workspace_id: str,
        paths: tuple[str, ...],
    ) -> list[WorkspaceDirectoryRecord]: ...


class WorkspaceDirectoryService:
    def __init__(self, *, repository: WorkspaceDirectoryRepository) -> None:
        self._repository = repository

    async def list_directories(self, *, workspace_id: str) -> list[WorkspaceDirectoryRecord]:
        return await self._repository.list_directories(workspace_id=workspace_id)

    async def create(
        self,
        *,
        workspace_id: str,
        path: str,
    ) -> list[WorkspaceDirectoryRecord]:
        normalized_path = normalize_workspace_directory_path(path)
        return await self._repository.ensure_directories(
            workspace_id=workspace_id,
            paths=_parent_paths(normalized_path),
        )


def normalize_workspace_directory_path(path: str) -> str:
    """Return a canonical logical path, rejecting paths that can escape a workspace."""

    normalized = path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("directory path must not be blank")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("directory path must be relative to the workspace")
    if "\x00" in normalized:
        raise ValueError("directory path must not contain a null character")

    parts: list[str] = []
    for part in normalized.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("directory path must not contain parent traversal")
        if not part.strip() or any(ord(character) < 32 for character in part):
            raise ValueError("directory path contains an invalid segment")
        parts.append(part)
    if not parts:
        raise ValueError("directory path must not be blank")

    result = "/".join(parts)
    if len(result) > 512:
        raise ValueError("directory path must be at most 512 characters")
    return result


def _parent_paths(path: str) -> tuple[str, ...]:
    parts = path.split("/")
    return tuple("/".join(parts[:index]) for index in range(1, len(parts) + 1))
