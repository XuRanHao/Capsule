from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import get_settings
from capsule.db.models import ProcessingJob, Workspace
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deleting_one_workspace_preserves_other_workspace(tmp_path: Path) -> None:
    database = Database(get_settings())
    suffix = uuid4().hex[:12]
    selected_id = f"workspace_delete_{suffix}"
    other_id = f"workspace_keep_{suffix}"
    repository = AssetRepository(database)
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        await repository.create_workspace(name="Delete me", workspace_id=selected_id)
        await repository.create_workspace(name="Keep me", workspace_id=other_id)
        selected_job = await repository.create_job(
            workspace_id=selected_id,
            input_path=tmp_path / "selected",
            total_count=1,
        )
        other_job = await repository.create_job(
            workspace_id=other_id,
            input_path=tmp_path / "other",
            total_count=1,
        )

        snapshot = await repository.delete_workspace_records(workspace_id=selected_id)

        assert snapshot.workspace_id == selected_id
        assert snapshot.job_count == 1
        async with database.session() as session:
            assert await session.get(Workspace, selected_id) is None
            assert await session.get(ProcessingJob, selected_job) is None
            assert await session.get(Workspace, other_id) is not None
            assert await session.get(ProcessingJob, other_job) is not None
    finally:
        try:
            async with database.session() as session, session.begin():
                await session.execute(
                    delete(Workspace).where(
                        Workspace.workspace_id.in_([selected_id, other_id])
                    )
                )
        finally:
            await database.dispose()
