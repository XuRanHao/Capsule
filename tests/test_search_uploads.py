from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from capsule.search.uploads import QueryImageNotFoundError, QueryImageService


class FakeDatabase:
    def __init__(self, record: object | None) -> None:
        self._record = record

    @asynccontextmanager
    async def session(self):
        record = self._record

        class Session:
            async def scalar(self, _statement):
                return record

        yield Session()


class FakeStorage:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.downloaded_keys: list[str] = []

    async def download_object(self, object_key: str) -> bytes:
        self.downloaded_keys.append(object_key)
        return self.content


@pytest.mark.asyncio
async def test_resolve_returns_inline_image_data() -> None:
    storage = FakeStorage(b"png-content")
    service = QueryImageService(
        FakeDatabase(
            SimpleNamespace(
                object_key="query-images/workspace_demo/query.png",
                content_type="image/png",
            )
        ),
        storage,
    )

    resolved = await service.resolve(
        workspace_id="workspace_demo",
        upload_id="query_image_demo",
    )

    assert resolved == "data:image/png;base64,cG5nLWNvbnRlbnQ="
    assert storage.downloaded_keys == ["query-images/workspace_demo/query.png"]


@pytest.mark.asyncio
async def test_resolve_rejects_unknown_upload() -> None:
    service = QueryImageService(FakeDatabase(None), FakeStorage(b"unused"))

    with pytest.raises(QueryImageNotFoundError):
        await service.resolve(
            workspace_id="workspace_demo",
            upload_id="query_image_missing",
        )
