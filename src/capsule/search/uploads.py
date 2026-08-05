import base64
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from capsule.db.models import QueryImageUpload
from capsule.db.session import Database
from capsule.storage.object_storage import ObjectStorage


class QueryImageNotFoundError(LookupError):
    pass


class QueryImageService:
    def __init__(self, database: Database, storage: ObjectStorage) -> None:
        self._database = database
        self._storage = storage

    async def upload(
        self,
        *,
        workspace_id: str,
        content: bytes,
        content_type: str,
        original_file_name: str,
    ) -> tuple[str, str]:
        upload_id = f"query_image_{uuid4().hex}"
        suffix = Path(original_file_name).suffix.lower()[:12] or ".bin"
        object_key = f"query-images/{workspace_id}/{upload_id}{suffix}"
        await self._storage.upload_bytes(
            content,
            object_key,
            content_type=content_type,
        )
        async with self._database.session() as session:
            session.add(
                QueryImageUpload(
                    upload_id=upload_id,
                    workspace_id=workspace_id,
                    object_key=object_key,
                    content_type=content_type,
                    file_size_bytes=len(content),
                )
            )
            await session.commit()
        return upload_id, await self._storage.presigned_get_url(object_key)

    async def resolve(
        self,
        *,
        workspace_id: str,
        upload_id: str,
    ) -> str:
        async with self._database.session() as session:
            record = await session.scalar(
                select(QueryImageUpload).where(
                    QueryImageUpload.upload_id == upload_id,
                    QueryImageUpload.workspace_id == workspace_id,
                )
            )
        if record is None:
            raise QueryImageNotFoundError(upload_id)
        content = await self._storage.download_object(record.object_key)
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{record.content_type};base64,{encoded}"
