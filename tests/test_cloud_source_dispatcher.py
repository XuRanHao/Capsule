"""Cloud source rows become durable tasks without an upload-service dependency."""

from __future__ import annotations

from types import SimpleNamespace

from capsule.db.models import ProcessingJob
from capsule.db.video_tasks import VideoProcessingTask
from capsule.pipeline.cloud_source_dispatcher import CloudSourceTaskDispatcher


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _Session:
    def __init__(self, source: object) -> None:
        self._source = source
        self.added: list[object] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> _Session:
        return self

    async def scalars(self, _statement: object) -> _ScalarRows:
        return _ScalarRows([self._source])

    async def scalar(self, _statement: object) -> None:
        return None

    def add(self, row: object) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        return None


class _Database:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def session(self) -> _Session:
        return self._session


async def test_dispatcher_creates_one_task_from_one_pending_s3_source() -> None:
    source = SimpleNamespace(
        source_file_id="src-1",
        workspace_id="workspace-1",
        relative_path="documents/notes.md",
        storage_uri="s3://capsule/sources/notes.md",
        processing_status="pending",
        processing_generation=3,
        sha256="a" * 64,
        processing_fingerprint="fingerprint-1",
        error_message="old error",
    )
    session = _Session(source)
    dispatcher = CloudSourceTaskDispatcher(database=_Database(session))  # type: ignore[arg-type]

    created = await dispatcher.run_once()

    assert created == 1
    assert source.processing_status == "processing"
    assert source.error_message is None
    job = next(row for row in session.added if isinstance(row, ProcessingJob))
    task = next(row for row in session.added if isinstance(row, VideoProcessingTask))
    assert job.workspace_id == "workspace-1"
    assert job.input_path == "s3://capsule/sources/notes.md"
    assert task.parent_job_id == job.job_id
    assert task.source_file_id == "src-1"
    assert task.source_generation == 3
    assert task.task_kind == "text"
    assert task.route_key == "cpu_text"
