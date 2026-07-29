from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import Settings, get_settings
from capsule.db.models import EmbeddingRecord, Workspace
from capsule.db.repositories import AssetRepository, EmbeddingAsset, EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import AssetType, EmbeddingStatus, EmbeddingType
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.schemas import AssetDraft, DiscoveredFile, EmbeddingResult
from capsule.vectorstore.milvus import VectorRecord


class FakeEmbeddingClient:
    def __init__(self, *, fail_text: str | None = None) -> None:
        self.fail_text = fail_text
        self.inputs: list[list[dict[str, Any]]] = []

    async def embed_multimodal(self, input_items: list[dict[str, Any]]) -> EmbeddingResult:
        captured = [dict(item) for item in input_items]
        self.inputs.append(captured)
        if self.fail_text and any(item.get("text") == self.fail_text for item in input_items):
            raise RuntimeError("intentional embedding failure")
        return EmbeddingResult(
            vector=[0.0, 1.0, 2.0],
            model="fake-seed-embedding",
            usage={"total_tokens": 3},
        )


class FakeVectorStore:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.records: list[VectorRecord] = []

    async def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return self.ensure_calls == 1

    async def aupsert(self, records: list[VectorRecord]) -> None:
        self.records.extend(records)


class FakeVideoUrlSigner:
    def __init__(self) -> None:
        self.uris: list[str] = []

    async def presigned_get_uri(self, uri: str, *, expires_seconds: int = 3600) -> str:
        self.uris.append(uri)
        return "https://objects.example.test/segment.mp4?signature=temporary"


def _settings(database_url: str) -> Settings:
    return Settings(
        database_url=database_url,
        ark_api_key=SecretStr("test-key"),
        embedding_dimension=3,
        embedding_concurrency=2,
    )


async def _seed_markdown_assets(
    *,
    database: Database,
    tmp_path: Path,
    workspace_id: str,
    contents: list[str],
) -> list[str]:
    path = tmp_path / "notes.md"
    path.write_text("\n\n".join(contents), encoding="utf-8")
    discovered = DiscoveredFile(
        path=str(path),
        relative_path="notes.md",
        extension=".md",
        size_bytes=path.stat().st_size,
    )
    repository = AssetRepository(database)
    source_file_id = await repository.get_or_create_source_file(
        workspace_id=workspace_id,
        source_file=discovered,
        sha256=sha256_file(path),
        mime_type="text/markdown",
    )
    assets = AssetFactory().build_many(
        workspace_id=workspace_id,
        source_file_id=source_file_id,
        source_sha256=sha256_file(path),
        source_file=discovered,
        drafts=[
            AssetDraft(
                asset_type=AssetType.MARKDOWN_BLOCK,
                file_name=path.name,
                source_locator={
                    "type": "text_range",
                    "block_index": index,
                    "char_start": 0,
                    "char_end": len(content),
                },
                raw_content=content,
            )
            for index, content in enumerate(contents)
        ],
    )
    result = await repository.replace_assets(source_file_id=source_file_id, assets=assets)
    return result.asset_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_service_persists_and_reuses_native_vectors(tmp_path: Path) -> None:
    base_settings = get_settings()
    database = Database(base_settings)
    workspace_id = f"workspace_embedding_{uuid4().hex[:12]}"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        asset_ids = await _seed_markdown_assets(
            database=database,
            tmp_path=tmp_path,
            workspace_id=workspace_id,
            contents=["# First\n\nA blue sky", "# Second\n\nA green hill"],
        )
        client = FakeEmbeddingClient()
        vectors = FakeVectorStore()
        service = AssetEmbeddingService(
            settings=_settings(base_settings.database_url),
            repository=EmbeddingRepository(database),
            model_client=client,
            vector_store=vectors,
        )

        first = await service.run(workspace_id=workspace_id)
        second = await service.run(workspace_id=workspace_id)
        forced = await service.run(workspace_id=workspace_id, force=True)

        assert first.indexed_count == 2
        assert first.skipped_count == 0
        assert first.failed_count == 0
        assert first.embedding_ids
        assert second.indexed_count == 0
        assert second.skipped_count == 2
        assert forced.indexed_count == 2
        assert len(client.inputs) == 4
        assert len(vectors.records) == 4
        assert {record.asset_id for record in vectors.records} == set(asset_ids)
        assert len({record.embedding_id for record in vectors.records}) == 2
        assert vectors.ensure_calls == 3

        async with database.session() as session:
            records = list(
                await session.scalars(
                    select(EmbeddingRecord).where(EmbeddingRecord.workspace_id == workspace_id)
                )
            )
        assert len(records) == 2
        assert all(record.status == EmbeddingStatus.INDEXED.value for record in records)
        assert all(record.dimension == 3 for record in records)
        assert all(
            record.embedding_type == EmbeddingType.NATIVE_MULTIMODAL.value
            for record in records
        )
        assert all(record.usage == {"total_tokens": 3} for record in records)
    finally:
        async with database.session() as session, session.begin():
            await session.execute(delete(Workspace).where(Workspace.workspace_id == workspace_id))
        await database.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embedding_service_records_failure_without_stopping_other_assets(
    tmp_path: Path,
) -> None:
    base_settings = get_settings()
    database = Database(base_settings)
    workspace_id = f"workspace_embedding_failure_{uuid4().hex[:12]}"
    try:
        try:
            async with database.session() as session:
                await session.execute(text("select 1"))
        except SQLAlchemyError:
            pytest.skip("PostgreSQL integration database is unavailable")

        asset_ids = await _seed_markdown_assets(
            database=database,
            tmp_path=tmp_path,
            workspace_id=workspace_id,
            contents=["good", "bad"],
        )
        service = AssetEmbeddingService(
            settings=_settings(base_settings.database_url),
            repository=EmbeddingRepository(database),
            model_client=FakeEmbeddingClient(fail_text="bad"),
            vector_store=FakeVectorStore(),
        )

        result = await service.run(workspace_id=workspace_id)

        assert result.indexed_count == 1
        assert result.failed_count == 1
        assert result.errors[0]["asset_id"] in asset_ids
        assert "intentional embedding failure" in result.errors[0]["error"]
        async with database.session() as session:
            records = list(
                await session.scalars(
                    select(EmbeddingRecord).where(EmbeddingRecord.workspace_id == workspace_id)
                )
            )
        assert {record.status for record in records} == {
            EmbeddingStatus.INDEXED.value,
            EmbeddingStatus.FAILED.value,
        }
    finally:
        async with database.session() as session, session.begin():
            await session.execute(delete(Workspace).where(Workspace.workspace_id == workspace_id))
        await database.dispose()


@pytest.mark.asyncio
async def test_embedding_inputs_use_original_image_bytes_and_playable_video_url(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "still.png"
    image_path.write_bytes(b"png-content")
    service = AssetEmbeddingService(
        settings=Settings(ark_api_key=SecretStr("test-key"), embedding_dimension=3),
        repository=cast(EmbeddingRepository, object()),
        model_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(),
        video_url_signer=FakeVideoUrlSigner(),
    )
    image = EmbeddingAsset(
        asset_id="asset_image",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="src_image",
        asset_type=AssetType.IMAGE.value,
        file_type=".png",
        content_hash="a" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri=image_path.as_uri(),
        source_mime_type="image/png",
    )
    video = EmbeddingAsset(
        asset_id="asset_video",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="src_video",
        asset_type=AssetType.VIDEO_SEGMENT.value,
        file_type=".mp4",
        content_hash="b" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri="s3://capsule/derived/video-segments/segment.mp4",
        source_storage_uri="file:///unused.mp4",
        source_mime_type="video/mp4",
    )

    image_input = await service._build_input(image, EmbeddingType.NATIVE_MULTIMODAL)
    video_input = await service._build_input(video, EmbeddingType.NATIVE_MULTIMODAL)

    assert image_input.input_items[0]["type"] == "image_url"
    assert image_input.input_items[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert video_input.input_items == [
        {
            "type": "video_url",
            "video_url": {
                "url": "https://objects.example.test/segment.mp4?signature=temporary"
            },
        }
    ]
