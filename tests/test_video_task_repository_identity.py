"""Schema-level contracts for the generic identity extension of video tasks."""

from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import (
    Column,
    ColumnDefault,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.pipeline.video_task_runtime import ProcessingTaskKind, VideoTaskLease, VideoTaskMessage


def test_task_table_keeps_video_defaults_and_kind_aware_identity() -> None:
    table = cast(Table, VideoProcessingTask.__table__)
    columns = table.c

    assert {
        "task_kind",
        "resource_class",
        "processor_version",
        "route_key",
        "lease_token",
    } <= set(
        columns.keys()
    )
    assert cast(ColumnDefault, columns.task_kind.default).arg == "video"
    assert cast(ColumnDefault, columns.resource_class.default).arg == "mps_video"
    assert cast(ColumnDefault, columns.route_key.default).arg == "mps_video"
    assert cast(ColumnDefault, columns.processor_version.default).arg == 1

    unique_columns = {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
        and constraint.name == "uq_processing_task_source_generation_result_version"
    }
    assert unique_columns == {
        ("source_file_id", "source_generation", "result_version")
    }
    indexes = {index.name for index in table.indexes}
    assert "ix_processing_tasks_kind_status_next_retry" in indexes
    assert "ix_processing_tasks_kind_identity" in indexes


def test_video_repository_binds_every_fence_to_complete_default_identity() -> None:
    repository = PostgresVideoTaskRepository(cast(async_sessionmaker[AsyncSession], None))

    identity = " AND ".join(str(clause) for clause in repository._task_identity_clauses())

    assert "processing_tasks.task_kind" in identity
    assert "processing_tasks.resource_class" in identity
    assert "processing_tasks.route_key" in identity
    assert "processing_tasks.processor_version" in identity


def test_message_identity_guard_rejects_another_processing_kind() -> None:
    repository = PostgresVideoTaskRepository(cast(async_sessionmaker[AsyncSession], None))
    message = VideoTaskMessage(
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
        task_kind=ProcessingTaskKind.TEXT,
    )

    assert not repository._message_matches_identity(message)


@pytest.mark.asyncio
async def test_inspect_message_contract_compares_the_full_persisted_identity() -> None:
    task = VideoProcessingTask(
        task_id="task-1",
        parent_job_id="job-1",
        source_file_id="source-1",
        source_generation=3,
        result_version=2,
        task_kind="video",
        resource_class="mps_video",
        route_key="mps_video",
        processor_version=4,
    )

    class _Result:
        def __init__(self, row: tuple[VideoProcessingTask, str, str] | None) -> None:
            self._row = row

        def one_or_none(self) -> tuple[VideoProcessingTask, str, str] | None:
            return self._row

    class _Session:
        def __init__(self, row: tuple[VideoProcessingTask, str, str] | None) -> None:
            self._row = row

        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, _statement: Any) -> _Result:
            return _Result(self._row)

    message = VideoTaskMessage(
        task_id="task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=3,
        result_version=2,
        processor_version=4,
        source_uri="file:///imports/demo%20video.mp4",
    )
    repository = PostgresVideoTaskRepository(
        cast(
            async_sessionmaker[AsyncSession],
            lambda: _Session((task, "workspace-1", "FILE:///imports/demo%20video.mp4")),
        ),
        processor_version=4,
    )

    assert await repository.inspect_message_contract(message)
    assert not await repository.inspect_message_contract(
        replace(message, workspace_id="wrong-workspace")
    )
    assert not await repository.inspect_message_contract(
        replace(message, source_uri="file:///private/imports/other.mp4")
    )

    missing = PostgresVideoTaskRepository(
        cast(async_sessionmaker[AsyncSession], lambda: _Session(None)),
        processor_version=4,
    )
    assert await missing.inspect_message_contract(message) is None


@pytest.mark.asyncio
async def test_can_ack_unclaimed_cleans_a_task_deleted_after_contract_validation() -> None:
    class _Result:
        def one_or_none(self) -> None:
            return None

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def execute(self, _statement: Any) -> _Result:
            return _Result()

    repository = PostgresVideoTaskRepository(
        cast(async_sessionmaker[AsyncSession], lambda: _Session())
    )
    message = VideoTaskMessage(
        task_id="orphan-task",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
    )

    assert await repository.can_ack_unclaimed(message)


def test_video_repository_rejects_an_invalid_processor_identity() -> None:
    with pytest.raises(ValueError, match="processor_version"):
        PostgresVideoTaskRepository(
            cast(async_sessionmaker[AsyncSession], None), processor_version=0
        )


def test_lease_token_defaults_empty_for_old_construction_only() -> None:
    lease = VideoTaskLease(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
    )

    assert lease.lease_token == ""


def test_nonempty_lease_token_is_part_of_the_repository_write_fence() -> None:
    repository = PostgresVideoTaskRepository(
        cast(async_sessionmaker[AsyncSession], None),
        task_kind="text",
        resource_class="cpu",
        route_key="text_cpu",
        processor_version=9,
    )
    lease = VideoTaskLease(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
        processor_version=7,
        lease_token="claim-unique-token",
    )

    clauses = " AND ".join(str(clause) for clause in repository._fence_clauses(lease))

    assert "processing_tasks.lease_token" in clauses
    assert "processing_tasks.dispatch_round" not in clauses
    identity = " AND ".join(
        str(clause) for clause in repository._lease_identity_clauses(lease)
    )
    assert all(
        column in identity
        for column in (
            "processing_tasks.task_kind",
            "processing_tasks.resource_class",
            "processing_tasks.route_key",
            "processing_tasks.processor_version",
        )
    )


@pytest.mark.asyncio
async def test_claim_lease_uses_the_fenced_message_and_repository_identity() -> None:
    class _Result:
        rowcount = 1

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        def begin(self) -> _Session:
            return self

        async def execute(self, _statement: Any) -> _Result:
            return _Result()

    repository = PostgresVideoTaskRepository(
        cast(async_sessionmaker[AsyncSession], lambda: _Session()),
        processor_version=7,
    )
    message = VideoTaskMessage(
        task_id="task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
        processor_version=7,
    )

    lease = await repository.claim_attempt(message, worker_id="worker-1", receipt="1-0")
    reclaimed_lease = await repository.claim_attempt(
        message, worker_id="worker-2", receipt="2-0"
    )

    assert lease is not None
    assert reclaimed_lease is not None
    assert lease.task_kind is ProcessingTaskKind.VIDEO
    assert lease.resource_class.value == "mps_video"
    assert lease.route_key == "mps_video"
    assert lease.processor_version == 7
    assert lease.worker_id == "worker-1"
    assert lease.result_version == 1
    assert lease.lease_token
    assert lease.lease_token.startswith("1:")
    assert reclaimed_lease.lease_token
    assert reclaimed_lease.lease_token != lease.lease_token


def test_empty_lease_token_cannot_form_a_valid_repository_write_fence() -> None:
    repository = PostgresVideoTaskRepository(cast(async_sessionmaker[AsyncSession], None))
    lease = VideoTaskLease(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
    )

    clauses = " AND ".join(str(clause) for clause in repository._fence_clauses(lease))

    assert "false" in clauses


@pytest.mark.asyncio
async def test_repository_rejects_an_empty_lease_token_before_writing() -> None:
    repository = PostgresVideoTaskRepository(cast(async_sessionmaker[AsyncSession], None))
    lease = VideoTaskLease(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        attempt=1,
        worker_id="worker-1",
        result_version=1,
    )

    assert not await repository.heartbeat(lease)
    assert not await repository.fail(lease, error="empty lease token")


def test_recovery_fence_compares_a_persisted_null_token_exactly() -> None:
    repository = PostgresVideoTaskRepository(cast(async_sessionmaker[AsyncSession], None))
    recovered_row = VideoProcessingTask(
        task_id="task-1",
        source_file_id="source-1",
        source_generation=1,
        result_version=1,
        attempt=1,
        owner_id="worker-1",
        lease_token=None,
    )

    clauses = " AND ".join(str(clause) for clause in repository._fence_clauses(recovered_row))

    assert "processing_tasks.lease_token IS NULL" in clauses


def test_0013_upgrade_keeps_the_existing_unique_constraint_and_task_lifecycle_columns() -> None:
    migration_path = (
        Path(__file__).parents[1]
        / "migrations"
        / "versions"
        / "20260812_0013_processing_task_identity.py"
    )
    namespace = runpy.run_path(str(migration_path))
    engine = create_engine("sqlite://")
    metadata = MetaData()
    old_tasks = Table(
        "video_processing_tasks",
        metadata,
        Column("task_id", String(64), primary_key=True),
        Column("source_file_id", String(64), nullable=False),
        Column("source_generation", Integer, nullable=False),
        Column("result_version", Integer, nullable=False),
        Column("status", String(32), nullable=False),
        Column("stage", String(32), nullable=False),
        Column("next_retry_at", String(64), nullable=True),
        UniqueConstraint(
            "source_file_id",
            "source_generation",
            "result_version",
            name="uq_video_task_source_generation_result_version",
        ),
    )
    Table(
        "processing_jobs",
        metadata,
        Column("job_id", String(64), primary_key=True),
    )
    lifecycle_states = (
        "queued",
        "processing",
        "retry_wait",
        "result_committed",
        "completed",
        "failed",
    )
    with engine.begin() as connection:
        metadata.create_all(connection)
        connection.execute(
            old_tasks.insert(),
            [
                {
                    "task_id": f"task-{status}",
                    "source_file_id": f"source-{status}",
                    "source_generation": 1,
                    "result_version": 1,
                    "status": status,
                    "stage": status,
                }
                for status in lifecycle_states
            ],
        )
        namespace["upgrade"].__globals__["op"] = Operations(
            MigrationContext.configure(connection)
        )
        namespace["upgrade"]()

        upgraded = Table("video_processing_tasks", MetaData(), autoload_with=connection)
        columns = set(upgraded.c.keys())
        rows = connection.execute(
            select(upgraded.c.task_id, upgraded.c.status, upgraded.c.stage)
        ).all()
        unique_constraints = inspect(connection).get_unique_constraints("video_processing_tasks")

    assert {
        "task_kind",
        "resource_class",
        "processor_version",
        "route_key",
        "lease_token",
        "input_payload",
    } <= columns
    assert {
        tuple(constraint["column_names"])
        for constraint in unique_constraints
        if constraint["name"] == "uq_video_task_source_generation_result_version"
    } == {("source_file_id", "source_generation", "result_version")}
    assert {(row.status, row.stage) for row in rows} == {
        (status, status) for status in lifecycle_states
    }
