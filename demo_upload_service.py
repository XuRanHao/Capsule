"""Minimal standalone uploader for demos; it is not part of the Capsule API app.

Run separately with:
    uv run uvicorn demo_upload_service:app --port 8010
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from capsule.config import get_settings
from capsule.db.models import SourceFile, Workspace
from capsule.db.session import Database
from capsule.enums import ProcessingStatus
from capsule.storage.object_storage import ObjectStorage

app = FastAPI(title="Capsule Demo Upload Service")


@app.post("/uploads", status_code=status.HTTP_201_CREATED)
async def upload_source(
    workspace_id: str = Form(min_length=1, max_length=64),
    relative_path: str = Form(min_length=1),
    file: UploadFile = File(),
) -> dict[str, object]:
    """Upload one demo file and register its private S3 URI as pending work."""
    settings = get_settings()
    normalized_path = _relative_path(relative_path)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="empty file")
    if len(content) > settings.import_file_max_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file too large")

    suffix = PurePosixPath(normalized_path).suffix.lower()
    if not suffix:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="relative_path must include a file extension",
        )
    digest = sha256(content).hexdigest()
    storage = ObjectStorage(settings)
    await storage.ensure_bucket()
    storage_uri = await storage.upload_bytes(
        content,
        f"demo-sources/{workspace_id}/{digest}{suffix}",
        content_type=file.content_type or "application/octet-stream",
    )

    database = Database(settings)
    try:
        async with database.session() as session, session.begin():
            workspace = await session.get(Workspace, workspace_id)
            if workspace is None:
                session.add(Workspace(workspace_id=workspace_id, name=workspace_id))
                await session.flush()
            source = await session.scalar(
                select(SourceFile)
                .where(
                    SourceFile.workspace_id == workspace_id,
                    SourceFile.relative_path == normalized_path,
                )
                .with_for_update()
            )
            values = {
                "original_file_name": PurePosixPath(normalized_path).name,
                "file_type": suffix,
                "mime_type": file.content_type or "application/octet-stream",
                "storage_uri": storage_uri,
                "sha256": digest,
                "processing_fingerprint": "demo-upload-v1",
                "file_size_bytes": len(content),
                "processing_status": ProcessingStatus.PENDING.value,
                "error_message": None,
                "file_tree_context": list(PurePosixPath(normalized_path).parent.parts)
                if PurePosixPath(normalized_path).parent != PurePosixPath(".")
                else [],
            }
            if source is None:
                source = SourceFile(
                    workspace_id=workspace_id,
                    relative_path=normalized_path,
                    processing_generation=1,
                    **values,
                )
                session.add(source)
            else:
                for field, value in values.items():
                    setattr(source, field, value)
                source.processing_generation += 1
            await session.flush()
            return {
                "source_file_id": source.source_file_id,
                "storage_uri": source.storage_uri,
                "generation": source.processing_generation,
                "processing_status": source.processing_status,
            }
    finally:
        await database.dispose()


def _relative_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="relative_path must be a safe relative path",
        )
    return str(path)
