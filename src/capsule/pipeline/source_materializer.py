"""Worker-local materialization of trusted S3-compatible source objects."""

import asyncio
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capsule.schemas import DiscoveredFile
from capsule.storage.object_storage import ObjectStorage


@dataclass(frozen=True, slots=True)
class MaterializedSource:
    """A source object downloaded for one task, with mandatory cleanup."""

    source_file: DiscoveredFile
    sha256: str
    cleanup: Callable[[], Awaitable[None]]


async def load_postgres_object_source(
    session_factory: Any,
    storage: ObjectStorage,
    message: Any,
) -> MaterializedSource:
    """Resolve task identity from PostgreSQL and materialize only its current object."""
    from sqlalchemy import select

    from capsule.db.models import SourceFile

    async with session_factory() as session:
        source = await session.scalar(
            select(SourceFile).where(SourceFile.source_file_id == message.source_file_id)
        )
    if (
        source is None
        or source.workspace_id != message.workspace_id
        or source.processing_generation != message.generation
    ):
        raise ValueError("task source is not the current canonical generation")
    if not source.storage_uri.startswith("s3://"):
        raise ValueError("task source storage must be an s3:// URI")

    suffix = Path(source.relative_path).suffix.lower()
    directory = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="capsule-source-"))
    local_path = directory / f"source{suffix}"
    try:
        await storage.download_uri_to_file(source.storage_uri, local_path)
    except BaseException:
        await asyncio.to_thread(shutil.rmtree, directory, True)
        raise

    async def cleanup() -> None:
        await asyncio.to_thread(shutil.rmtree, directory, True)

    return MaterializedSource(
        source_file=DiscoveredFile(
            path=str(local_path),
            relative_path=source.relative_path,
            extension=suffix,
            size_bytes=source.file_size_bytes,
        ),
        sha256=source.sha256,
        cleanup=cleanup,
    )
