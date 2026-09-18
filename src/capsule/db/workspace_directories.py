"""Persistence for user-created workspace directories."""

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.db.models import Workspace, WorkspaceDirectory
from capsule.db.session import Database
from capsule.schemas import WorkspaceDirectoryRecord


class WorkspaceDirectoryWorkspaceNotFoundError(ValueError):
    """The requested workspace does not exist."""


class WorkspaceDirectoryRepository:
    """Store the explicit directory skeleton beside imported asset paths."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def list_directories(self, *, workspace_id: str) -> list[WorkspaceDirectoryRecord]:
        async with self._database.session() as session:
            await self._require_workspace(session, workspace_id=workspace_id)
            rows = list(
                await session.scalars(
                    select(WorkspaceDirectory)
                    .where(WorkspaceDirectory.workspace_id == workspace_id)
                    .order_by(WorkspaceDirectory.path)
                )
            )
        return [_record(row) for row in rows]

    async def ensure_directories(
        self,
        *,
        workspace_id: str,
        paths: tuple[str, ...],
    ) -> list[WorkspaceDirectoryRecord]:
        """Create paths if absent and return every requested parent-to-leaf path."""

        async with self._database.session() as session, session.begin():
            await self._require_workspace(session, workspace_id=workspace_id, lock=True)
            await session.execute(
                pg_insert(WorkspaceDirectory)
                .values([{"workspace_id": workspace_id, "path": path} for path in paths])
                .on_conflict_do_nothing(index_elements=["workspace_id", "path"])
            )
            rows = list(
                await session.scalars(
                    select(WorkspaceDirectory)
                    .where(
                        WorkspaceDirectory.workspace_id == workspace_id,
                        WorkspaceDirectory.path.in_(paths),
                    )
                    .order_by(WorkspaceDirectory.path)
                )
            )
        records_by_path = {row.path: _record(row) for row in rows}
        return [records_by_path[path] for path in paths]

    @staticmethod
    async def _require_workspace(
        session: AsyncSession,
        *,
        workspace_id: str,
        lock: bool = False,
    ) -> None:
        # `AsyncSession.get` accepts `with_for_update`; keeping the lock for
        # creation serializes directory writes with workspace deletion.
        workspace = await session.get(
            Workspace,
            workspace_id,
            with_for_update=lock,
        )
        if workspace is None:
            raise WorkspaceDirectoryWorkspaceNotFoundError(
                f"workspace does not exist: {workspace_id}"
            )


def _record(directory: WorkspaceDirectory) -> WorkspaceDirectoryRecord:
    return WorkspaceDirectoryRecord(
        path=directory.path,
        created_at=directory.created_at,
        updated_at=directory.updated_at,
    )
