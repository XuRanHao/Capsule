import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import delete, select, text
from sqlalchemy.exc import SQLAlchemyError

from capsule.config import Settings, get_settings
from capsule.db.models import EmbeddingRecord, Workspace
from capsule.db.repositories import (
    AssetRepository,
    ClusterEmbeddingAsset,
    EmbeddingAsset,
    EmbeddingRepository,
)
from capsule.db.session import Database
from capsule.enums import AssetType, EmbeddingStatus, EmbeddingType
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.embedding import AssetEmbeddingService, EmbeddingInputUnavailable
from capsule.schemas import AssetDraft, DiscoveredFile, EmbeddingResult
from capsule.vectorstore.milvus import VectorRecord


class FakeEmbeddingClient:
    def __init__(self, *, fail_text: str | None = None) -> None:
        self.fail_text = fail_text
        self.inputs: list[list[dict[str, Any]]] = []

    async def embed_multimodal(
        self,
        input_items: Sequence[Mapping[str, Any]],
    ) -> EmbeddingResult:
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
        self.fused_records: list[VectorRecord] = []
        self.raw_vectors: dict[str, list[float]] = {}
        self.fetch_calls = 0
        self.fetch_visibility_delay = 0
        self.fetch_requests: list[tuple[str, ...]] = []

    async def ensure_collection(self) -> bool:
        self.ensure_calls += 1
        return self.ensure_calls == 1

    async def aupsert(self, records: list[VectorRecord]) -> None:
        self.records.extend(records)

    async def aupsert_fused(self, records: list[VectorRecord]) -> None:
        self.fused_records.extend(records)

    async def fetch_vectors(
        self,
        embedding_ids: Sequence[str],
    ) -> dict[str, list[float]]:
        self.fetch_calls += 1
        self.fetch_requests.append(tuple(embedding_ids))
        if self.fetch_calls <= self.fetch_visibility_delay:
            return {}
        return {
            embedding_id: self.raw_vectors[embedding_id]
            for embedding_id in embedding_ids
            if embedding_id in self.raw_vectors
        }

    async def fetch_fused_vector_ids(
        self,
        embedding_ids: Sequence[str],
    ) -> set[str]:
        requested = set(embedding_ids)
        return {
            record.embedding_id
            for record in self.fused_records
            if record.embedding_id in requested
        }


class FakeArtifactReader:
    def __init__(self) -> None:
        self.uris: list[str] = []

    async def download_uri(self, uri: str) -> bytes:
        self.uris.append(uri)
        return b"fake-mp4"


class _MutableSearchVectorRepository:
    def __init__(self) -> None:
        self.revision = 1
        self.asset = EmbeddingAsset(
            asset_id="asset_1",
            workspace_id="workspace_demo",
            project_id="project_default",
            source_file_id="source_1",
            asset_type=AssetType.IMAGE.value,
            file_type=".png",
            content_hash="1" * 64,
            embedding_revision=3,
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
            raw_content=None,
            asset_description="测试图片",
            asset_features={"visual_style": {"value": "插画"}},
            derived_file_uri=None,
            source_storage_uri="file:///asset_1.png",
            source_mime_type="image/png",
        )

    async def list_assets(self, **_: object) -> list[EmbeddingAsset]:
        return [self.asset]

    async def list_indexed_cluster_embeddings(
        self,
        *,
        embedding_type: str,
        **_: object,
    ) -> list[ClusterEmbeddingAsset]:
        prefix = "native" if embedding_type == "native_multimodal" else "visual"
        return [
            ClusterEmbeddingAsset(
                embedding_id=f"emb_{prefix}_{self.revision}",
                asset_id=self.asset.asset_id,
                source_file_id=self.asset.source_file_id,
                asset_type=self.asset.asset_type,
                asset_name=None,
                asset_description=self.asset.asset_description,
                asset_features=self.asset.asset_features,
                file_tree_context=[],
            )
        ]

    def raw_vectors(self) -> dict[str, list[float]]:
        return {
            f"emb_native_{self.revision}": [1.0, 0.0],
            f"emb_visual_{self.revision}": [0.0, 1.0],
        }


def _search_vector_service(
    *,
    repository: _MutableSearchVectorRepository,
    vectors: FakeVectorStore,
    visibility_timeout: float = 0,
) -> AssetEmbeddingService:
    return AssetEmbeddingService(
        settings=Settings(
            embedding_dimension=2,
            search_vector_visibility_timeout_seconds=visibility_timeout,
            search_vector_visibility_poll_initial_seconds=0.001,
            search_vector_visibility_poll_max_seconds=0.002,
        ),
        repository=cast(EmbeddingRepository, repository),
        model_client=FakeEmbeddingClient(),
        vector_store=vectors,
    )


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
        assert vectors.ensure_calls == 1

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
            record.embedding_type == EmbeddingType.NATIVE_MULTIMODAL.value for record in records
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
async def test_embedding_service_materializes_current_raw_pairs_into_fused_store() -> None:
    created_at = datetime(2026, 8, 10, tzinfo=UTC)
    assets = [
        EmbeddingAsset(
            asset_id=f"asset_{index}",
            workspace_id="workspace_demo",
            project_id="project_default",
            source_file_id=f"source_{index}",
            asset_type=AssetType.IMAGE.value,
            file_type=".png",
            content_hash=str(index) * 64,
            embedding_revision=3,
            created_at=created_at,
            raw_content=None,
            asset_description="测试图片",
            asset_features={"visual_style": {"value": "插画"}},
            derived_file_uri=None,
            source_storage_uri=f"file:///asset_{index}.png",
            source_mime_type="image/png",
        )
        for index in (1, 2)
    ]

    def indexed(embedding_id: str, asset_id: str) -> ClusterEmbeddingAsset:
        return ClusterEmbeddingAsset(
            embedding_id=embedding_id,
            asset_id=asset_id,
            source_file_id=asset_id.replace("asset", "source"),
            asset_type=AssetType.IMAGE.value,
            asset_name=None,
            asset_description="测试图片",
            asset_features={"visual_style": {"value": "插画"}},
            file_tree_context=[],
        )

    class Repository:
        async def list_assets(self, **_: object) -> list[EmbeddingAsset]:
            return assets

        async def list_indexed_cluster_embeddings(
            self,
            *,
            embedding_type: str,
            **_: object,
        ) -> list[ClusterEmbeddingAsset]:
            prefix = "native" if embedding_type == "native_multimodal" else "visual"
            return [indexed(f"emb_{prefix}_{index}", f"asset_{index}") for index in (1, 2)]

    vectors = FakeVectorStore()
    vectors.raw_vectors = {
        "emb_native_1": [1.0, 0.0],
        # The second native metadata row deliberately has no raw Milvus vector.
        "emb_visual_1": [0.0, 1.0],
        "emb_visual_2": [0.0, 1.0],
    }
    service = AssetEmbeddingService(
        settings=Settings(
            embedding_dimension=2,
            search_vector_visibility_timeout_seconds=0,
        ),
        repository=cast(EmbeddingRepository, Repository()),
        model_client=FakeEmbeddingClient(),
        vector_store=vectors,
    )

    result = await service.materialize_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[
            EmbeddingType.NATIVE_MULTIMODAL,
            EmbeddingType.VISUAL_STYLE,
        ],
        asset_ids=["asset_1", "asset_2"],
    )

    assert result.requested_count == 2
    assert result.materialized_count == 1
    assert result.embedding_ids == ("emb_visual_1",)
    assert result.failures[0].asset_id == "asset_2"
    assert "native_multimodal" in result.failures[0].reason
    assert len(vectors.fused_records) == 1
    fused = vectors.fused_records[0]
    assert fused.embedding_id == "emb_visual_1"
    assert fused.embedding_revision == 3
    assert fused.vector == pytest.approx(
        [0.3 / (0.58**0.5), 0.7 / (0.58**0.5)]
    )


@pytest.mark.asyncio
async def test_ensure_search_vectors_materializes_once_for_same_fingerprint() -> None:
    repository = _MutableSearchVectorRepository()
    vectors = FakeVectorStore()
    vectors.raw_vectors = repository.raw_vectors()
    service = _search_vector_service(repository=repository, vectors=vectors)

    first_results = await asyncio.gather(
        service.ensure_search_vectors(
            workspace_id="workspace_demo",
            embedding_types=[
                EmbeddingType.NATIVE_MULTIMODAL,
                EmbeddingType.VISUAL_STYLE,
            ],
        ),
        service.ensure_search_vectors(
            workspace_id="workspace_demo",
            embedding_types=[EmbeddingType.VISUAL_STYLE],
        ),
    )
    assert all(result == {} for result in first_results)
    assert len(vectors.fused_records) == 1

    assert await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    ) == {}
    assert len(vectors.fused_records) == 1

    restarted_service = _search_vector_service(repository=repository, vectors=vectors)
    assert await restarted_service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    ) == {}
    assert len(vectors.fused_records) == 1


@pytest.mark.asyncio
async def test_ensure_search_vectors_materializes_distinct_dimensions_in_parallel() -> None:
    repository = _MutableSearchVectorRepository()

    class ParallelFetchVectorStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.fetches_entered = 0
            self.both_entered = asyncio.Event()

        async def fetch_vectors(
            self,
            embedding_ids: Sequence[str],
        ) -> dict[str, list[float]]:
            self.fetches_entered += 1
            if self.fetches_entered == 2:
                self.both_entered.set()
            await asyncio.wait_for(self.both_entered.wait(), timeout=0.1)
            return await super().fetch_vectors(embedding_ids)

    vectors = ParallelFetchVectorStore()
    vectors.raw_vectors = repository.raw_vectors()
    service = _search_vector_service(repository=repository, vectors=vectors)

    assert await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[
            EmbeddingType.VISUAL_STYLE,
            EmbeddingType.MOOD_ATMOSPHERE,
        ],
    ) == {}
    assert len(vectors.fused_records) == 2


@pytest.mark.asyncio
async def test_ensure_search_vectors_invalidates_changed_source_fingerprint() -> None:
    repository = _MutableSearchVectorRepository()
    vectors = FakeVectorStore()
    vectors.raw_vectors = repository.raw_vectors()
    service = _search_vector_service(repository=repository, vectors=vectors)

    assert await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    ) == {}
    repository.revision = 2
    vectors.raw_vectors.update(repository.raw_vectors())

    assert await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    ) == {}
    assert [record.embedding_id for record in vectors.fused_records] == [
        "emb_visual_1",
        "emb_visual_2",
    ]


@pytest.mark.asyncio
async def test_ensure_search_vectors_waits_for_delayed_raw_vector_visibility() -> None:
    repository = _MutableSearchVectorRepository()
    vectors = FakeVectorStore()
    vectors.raw_vectors = repository.raw_vectors()
    vectors.fetch_visibility_delay = 2
    service = _search_vector_service(
        repository=repository,
        vectors=vectors,
        visibility_timeout=0.05,
    )

    errors = await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    )

    assert errors == {}
    assert vectors.fetch_calls == 3
    assert len(vectors.fused_records) == 1


@pytest.mark.asyncio
async def test_ensure_search_vectors_keeps_permanent_missing_failure_uncached() -> None:
    repository = _MutableSearchVectorRepository()
    vectors = FakeVectorStore()
    vectors.raw_vectors = repository.raw_vectors()
    vectors.fetch_visibility_delay = 100
    service = _search_vector_service(
        repository=repository,
        vectors=vectors,
        visibility_timeout=0.003,
    )

    first_errors = await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    )
    first_fetch_calls = vectors.fetch_calls
    second_errors = await service.ensure_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    )

    assert EmbeddingType.VISUAL_STYLE in first_errors
    assert "raw vector is missing" in first_errors[EmbeddingType.VISUAL_STYLE]
    assert EmbeddingType.VISUAL_STYLE in second_errors
    assert vectors.fetch_calls > first_fetch_calls
    assert vectors.fused_records == []


@pytest.mark.asyncio
async def test_materialization_does_not_wait_for_unrelated_native_vector() -> None:
    repository = _MutableSearchVectorRepository()
    unrelated_asset = replace(
        repository.asset,
        asset_id="asset_unrelated",
        source_file_id="source_unrelated",
    )

    class RepositoryWithUnrelatedNative(_MutableSearchVectorRepository):
        async def list_assets(self, **_: object) -> list[EmbeddingAsset]:
            return [self.asset, unrelated_asset]

        async def list_indexed_cluster_embeddings(
            self,
            *,
            embedding_type: str,
            **values: object,
        ) -> list[ClusterEmbeddingAsset]:
            records = await super().list_indexed_cluster_embeddings(
                embedding_type=embedding_type,
                **values,
            )
            if embedding_type == EmbeddingType.NATIVE_MULTIMODAL.value:
                records.append(
                    ClusterEmbeddingAsset(
                        embedding_id="emb_native_unrelated",
                        asset_id=unrelated_asset.asset_id,
                        source_file_id=unrelated_asset.source_file_id,
                        asset_type=unrelated_asset.asset_type,
                        asset_name=None,
                        asset_description=unrelated_asset.asset_description,
                        asset_features=unrelated_asset.asset_features,
                        file_tree_context=[],
                    )
                )
            return records

    specific_repository = RepositoryWithUnrelatedNative()
    vectors = FakeVectorStore()
    vectors.raw_vectors = specific_repository.raw_vectors()
    service = _search_vector_service(
        repository=specific_repository,
        vectors=vectors,
        visibility_timeout=0.05,
    )

    result = await service.materialize_search_vectors(
        workspace_id="workspace_demo",
        embedding_types=[EmbeddingType.VISUAL_STYLE],
    )

    assert result.failures == ()
    assert result.materialized_count == 1
    assert "emb_native_unrelated" not in {
        embedding_id
        for request in vectors.fetch_requests
        for embedding_id in request
    }


@pytest.mark.asyncio
async def test_embedding_inputs_use_image_and_video_data_uris(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "still.png"
    Image.new("RGB", (16, 12), "orange").save(image_path)
    service = AssetEmbeddingService(
        settings=Settings(ark_api_key=SecretStr("test-key"), embedding_dimension=3),
        repository=cast(EmbeddingRepository, object()),
        model_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(),
        artifact_reader=FakeArtifactReader(),
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
            "video_url": {"url": "data:video/mp4;base64,ZmFrZS1tcDQ="},
        }
    ]


    derived_image = replace(
        image,
        derived_file_uri=image_path.as_uri(),
        source_storage_uri="file:///container/document.docx",
        source_mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_info={"mime_type": "image/png", "embedded_in_document": True},
    )
    derived_input = await service._build_input(
        derived_image,
        EmbeddingType.NATIVE_MULTIMODAL,
    )
    assert derived_input.input_items[0]["image_url"]["url"].startswith("data:image/png;base64,")

    not_applicable = replace(
        image,
        asset_features={
            "character_state_or_psychology": {
                "value": "错误遗留的人物状态",
                "status": "not_applicable",
            }
        },
    )
    with pytest.raises(EmbeddingInputUnavailable):
        await service._build_input(
            not_applicable,
            EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY,
        )

    text_with_visual_metadata = replace(
        image,
        asset_type=AssetType.MARKDOWN_BLOCK.value,
        raw_content="一段带有模型视觉标签的文字。",
        asset_features={
            "visual_style": {"value": "赛博朋克", "status": "observed"}
        },
    )
    with pytest.raises(EmbeddingInputUnavailable, match="not supported"):
        await service._build_input(
            text_with_visual_metadata,
            EmbeddingType.VISUAL_STYLE,
        )

    usage = replace(
        image,
        source_relative_path="海报/素材/20251216-143446.png",
        asset_features={
            "asset_usage": {
                "value": "海报制作",
                "status": "metadata",
                "source_path": "海报/素材/20251216-143446.png",
            }
        },
    )
    usage_input = await service._build_input(usage, EmbeddingType.ASSET_USAGE)
    assert usage_input.input_items == [
        {
            "type": "text",
            "text": "素材用途：海报制作；来源目录：海报/素材",
        }
    ]


@pytest.mark.asyncio
async def test_logical_video_native_embedding_uses_transient_representative_frames() -> None:
    class FrameExtractor:
        def __init__(self) -> None:
            self.requests: list[object] = []

        async def extract(self, request) -> list[bytes]:
            self.requests.append(request)
            return [b"\xff\xd8first", b"\xff\xd8second"]

    extractor = FrameExtractor()
    service = AssetEmbeddingService(
        settings=Settings(ark_api_key=SecretStr("test-key"), embedding_dimension=3),
        repository=cast(EmbeddingRepository, object()),
        model_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(),
        video_frame_extractor=extractor,
    )
    video = EmbeddingAsset(
        asset_id="asset_logical_video",
        workspace_id="workspace",
        project_id="project_default",
        source_file_id="src_video",
        asset_type=AssetType.VIDEO_SEGMENT.value,
        file_type=".mov",
        content_hash="c" * 64,
        embedding_revision=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        raw_content=None,
        asset_description=None,
        asset_features={},
        derived_file_uri=None,
        source_storage_uri="file:///library/original.mov",
        source_mime_type="video/quicktime",
        file_info={
            "video_output_mode": "logical",
            "representative_frames": [
                {"timestamp_ms": 2_000},
                {"timestamp_ms": 4_000},
            ],
        },
        source_locator={"start_ms": 1_000, "end_ms": 5_000},
    )

    embedding_input = await service._build_input(
        video,
        EmbeddingType.NATIVE_MULTIMODAL,
    )

    assert [item["type"] for item in embedding_input.input_items] == [
        "image_url",
        "image_url",
    ]
    assert all(
        item["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for item in embedding_input.input_items
    )
    assert embedding_input.source_mode.value == "original_video"
    assert len(extractor.requests) == 1
    request = extractor.requests[0]
    assert request.source_uri == "file:///library/original.mov"
    assert request.timestamps_ms == (2_000, 4_000)


@pytest.mark.asyncio
async def test_native_video_embedding_has_a_dedicated_memory_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concurrency = 0
    maximum_concurrency = 0
    service = AssetEmbeddingService(
        settings=Settings(
            ark_api_key=SecretStr("test-key"),
            embedding_dimension=3,
            native_embedding_concurrency=24,
            video_native_embedding_concurrency=2,
        ),
        repository=cast(EmbeddingRepository, object()),
        model_client=FakeEmbeddingClient(),
        vector_store=FakeVectorStore(),
    )
    template = EmbeddingAsset(
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
        derived_file_uri="s3://capsule/segment.mp4",
        source_storage_uri="file:///unused.mp4",
        source_mime_type="video/mp4",
    )

    async def probe(**_: Any) -> None:
        nonlocal concurrency, maximum_concurrency
        concurrency += 1
        maximum_concurrency = max(maximum_concurrency, concurrency)
        await asyncio.sleep(0.01)
        concurrency -= 1

    monkeypatch.setattr(service, "_embed_one_with_shared_pool", probe)
    await asyncio.gather(
        *(
            service._embed_one(
                asset=replace(template, asset_id=f"asset_video_{index}"),
                embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
                force=False,
            )
            for index in range(8)
        )
    )

    assert maximum_concurrency == 2
