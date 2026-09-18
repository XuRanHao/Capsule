"""Canonical local and object-storage source materialization contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from capsule.db.models import SourceFile
from capsule.pipeline.source_materializer import load_postgres_object_source
from capsule.pipeline.video_task_runtime import VideoTaskMessage
from capsule.storage.object_storage import ObjectStorage


@pytest.mark.asyncio
async def test_materializer_accepts_a_current_local_import_source(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"image source")
    source = SourceFile(
        source_file_id="source-1",
        workspace_id="workspace-1",
        original_file_name=path.name,
        file_type="image",
        mime_type="image/jpeg",
        relative_path=path.name,
        storage_uri=path.as_uri(),
        sha256="a" * 64,
        processing_generation=1,
        file_size_bytes=path.stat().st_size,
    )

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def scalar(self, _statement: Any) -> SourceFile:
            return source

    materialized = await load_postgres_object_source(
        lambda: _Session(),
        cast(ObjectStorage, object()),
        VideoTaskMessage(
            task_id="task-1",
            job_id="job-1",
            workspace_id="workspace-1",
            source_file_id="source-1",
            generation=1,
        ),
    )

    assert materialized.source_file.path == str(path)
    assert materialized.source_file.extension == ".jpg"
    assert materialized.sha256 == source.sha256
    await materialized.cleanup()
