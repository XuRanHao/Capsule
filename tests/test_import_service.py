import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from capsule.config import Settings
from capsule.enums import EmbeddingType, PipelineStage
from capsule.pipeline.import_service import (
    BrowserImportService,
    ImportCompletion,
    enrich_assets,
)
from capsule.schemas import AssetUnderstanding, ProcessingJobRecord


class FakeAssetRepository:
    def __init__(self) -> None:
        self.import_root: Path | None = None
        self.status = "queued"
        self.started_count: int | None = None

    async def create_pending_import_job(self, **values: object) -> str:
        self.import_root = values["import_root"]  # type: ignore[assignment]
        return "job_import_test"

    async def get_job(self, *, job_id: str, workspace_id: str) -> ProcessingJobRecord:
        assert job_id == "job_import_test"
        assert workspace_id == "workspace_import_test"
        assert self.import_root is not None
        return ProcessingJobRecord(
            job_id=job_id,
            workspace_id=workspace_id,
            input_path=str(self.import_root / job_id),
            total_count=self.started_count or 0,
            completed_count=0,
            failed_count=0,
            status=self.status,
            current_stage="discovering",
            error_info=[],
            started_at=None,
            completed_at=None,
        )

    async def start_import_job(self, *, job_id: str, total_count: int) -> None:
        assert job_id == "job_import_test"
        self.status = "running"
        self.started_count = total_count

    async def fail_job(self, **_: object) -> None:
        raise AssertionError("import should not fail")


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def run(self, input_path: Path, workspace_id: str, *, job_id: str) -> SimpleNamespace:
        self.calls.append(
            {"input_path": input_path, "workspace_id": workspace_id, "job_id": job_id}
        )
        return SimpleNamespace(job_id=job_id)


@pytest.mark.asyncio
async def test_browser_import_uploads_each_file_before_assetization(tmp_path: Path) -> None:
    repository = FakeAssetRepository()
    runner = FakeRunner()
    service = BrowserImportService(
        settings=Settings(import_root=tmp_path / "imports"),
        repository=repository,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
    )

    job = await service.create_job(workspace_id="workspace_import_test")
    await service.upload_file(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
        file=UploadFile(filename="note.md", file=BytesIO(b"# First version")),
        relative_path="references/notes/note.md",
    )
    await service.upload_file(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
        file=UploadFile(filename="note.md", file=BytesIO(b"# Retried version")),
        relative_path="references/notes/note.md",
    )
    completion = await service.complete_job(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
    )
    await service.execute(completion=completion, workspace_id="workspace_import_test")

    assert (job.staged_path / "references/notes/note.md").read_bytes() == b"# Retried version"
    assert repository.started_count == 1
    assert runner.calls == [
        {
            "input_path": job.staged_path,
            "workspace_id": "workspace_import_test",
            "job_id": "job_import_test",
        }
    ]


@pytest.mark.asyncio
async def test_enrichment_runs_understanding_and_every_embedding_channel() -> None:
    class Repository:
        def __init__(self) -> None:
            self.stages: list[PipelineStage] = []
            self.final_errors: list[dict[str, str]] = []
            self.durations: dict[str, float] = {}

        async def begin_asset_enrichment(self, *, asset_ids: list[str]) -> None:
            assert asset_ids == ["asset_a", "asset_b"]

        async def set_job_stage(self, *, job_id: str, stage: PipelineStage) -> None:
            assert job_id == "job_test"
            self.stages.append(stage)

        async def add_job_stage_durations(
            self,
            *,
            job_id: str,
            durations_ms: dict[str, float],
        ) -> None:
            assert job_id == "job_test"
            self.durations = durations_ms

        async def finalize_enrichment(
            self,
            *,
            job_id: str,
            asset_ids: list[str],
            errors: list[dict[str, str]],
        ) -> None:
            assert job_id == "job_test"
            assert asset_ids == ["asset_a", "asset_b"]
            self.final_errors = errors

    native_started = asyncio.Event()

    class Understanding:
        async def run(self, **_: object) -> SimpleNamespace:
            await asyncio.wait_for(native_started.wait(), timeout=1)
            return SimpleNamespace(
                errors=[{"asset_id": "asset_b", "error": "understanding failed"}],
                understanding_duration_ms=120.0,
                feature_ready_duration_ms=5.0,
            )

    class Embedding:
        def __init__(self) -> None:
            self.types: list[EmbeddingType] = []

        async def run(
            self,
            *,
            embedding_type: EmbeddingType,
            **_: object,
        ) -> SimpleNamespace:
            self.types.append(embedding_type)
            native_started.set()
            return SimpleNamespace(
                embedding_type=embedding_type.value,
                errors=[],
                embedding_duration_ms=20.0,
                indexing_duration_ms=2.0,
            )

        async def run_many(
            self,
            *,
            embedding_types: list[EmbeddingType],
            **_: object,
        ) -> list[SimpleNamespace]:
            self.types.extend(embedding_types)
            return [
                SimpleNamespace(
                    embedding_type=embedding_type.value,
                    errors=[],
                    embedding_duration_ms=20.0,
                    indexing_duration_ms=2.0,
                )
                for embedding_type in embedding_types
            ]

    repository = Repository()
    embedding = Embedding()
    result = await enrich_assets(
        job_id="job_test",
        workspace_id="workspace_test",
        asset_ids=["asset_a", "asset_b"],
        repository=repository,  # type: ignore[arg-type]
        understanding_service=Understanding(),  # type: ignore[arg-type]
        embedding_service=embedding,  # type: ignore[arg-type]
    )

    assert repository.stages == [
        PipelineStage.UNDERSTANDING,
        PipelineStage.FEATURE_READY,
        PipelineStage.EMBEDDING,
        PipelineStage.INDEXING,
    ]
    assert embedding.types == list(EmbeddingType)
    assert result.completed_asset_count == 1
    assert result.partial_failed_asset_count == 1
    assert repository.final_errors[0]["stage"] == "understanding"
    assert repository.durations["understanding"] == 120.0
    assert repository.durations["feature_ready"] == 5.0
    assert repository.durations["embedding"] > 0
    assert repository.durations["indexing"] > 0
    assert repository.durations["embedding"] / repository.durations["indexing"] == pytest.approx(10)


@pytest.mark.asyncio
async def test_browser_import_enriches_asset_before_runner_finishes(tmp_path: Path) -> None:
    events: list[str] = []
    understanding_started = asyncio.Event()

    class Repository:
        def __init__(self) -> None:
            self.final_asset_ids: list[str] = []
            self.failures: list[str] = []

        async def set_job_stage(self, *, job_id: str, stage: PipelineStage) -> None:
            assert job_id == "job_stream"
            events.append(f"stage:{stage.value}")

        async def begin_asset_enrichment(self, *, asset_ids: list[str]) -> None:
            assert asset_ids == ["asset_a"]

        async def add_job_stage_durations(self, **_: object) -> None:
            return None

        async def finalize_enrichment(
            self,
            *,
            job_id: str,
            asset_ids: list[str],
            errors: list[dict[str, str]],
        ) -> None:
            assert job_id == "job_stream"
            assert errors == []
            self.final_asset_ids = asset_ids
            events.append("job_finalized")

        async def finalize_job(self, **_: object) -> None:
            raise AssertionError("an Asset Job must be finalized after enrichment")

        async def fail_job(self, *, error: str, **_: object) -> None:
            self.failures.append(error)

    class Runner:
        async def run(
            self,
            _input_path: Path,
            _workspace_id: str,
            *,
            job_id: str,
            on_assets_stored,
            finalize_job: bool,
        ) -> SimpleNamespace:
            assert job_id == "job_stream"
            assert not finalize_job
            events.append("asset_committed")
            await on_assets_stored(["asset_a"])
            await asyncio.wait_for(understanding_started.wait(), timeout=1)
            events.append("runner_finished")
            return SimpleNamespace(
                asset_ids=["asset_parent", "asset_a"],
                indexable_asset_ids=["asset_a"],
            )

    class Understanding:
        async def run(self, *, asset_ids: list[str], **_: object) -> SimpleNamespace:
            assert asset_ids == ["asset_a"]
            events.append("understanding_started")
            understanding_started.set()
            await asyncio.sleep(0)
            return SimpleNamespace(
                errors=[],
                understanding_duration_ms=10.0,
                feature_ready_duration_ms=1.0,
            )

    class Embedding:
        async def run(
            self,
            *,
            embedding_type: EmbeddingType,
            **_: object,
        ) -> SimpleNamespace:
            assert embedding_type is EmbeddingType.NATIVE_MULTIMODAL
            events.append("native_embedding_started")
            return SimpleNamespace(
                embedding_type=embedding_type.value,
                errors=[],
                embedding_duration_ms=2.0,
                indexing_duration_ms=1.0,
            )

        async def run_many(
            self,
            *,
            embedding_types: list[EmbeddingType],
            **_: object,
        ) -> list[SimpleNamespace]:
            assert understanding_started.is_set()
            events.append("text_embedding_started")
            return [
                SimpleNamespace(
                    embedding_type=embedding_type.value,
                    errors=[],
                    embedding_duration_ms=2.0,
                    indexing_duration_ms=1.0,
                )
                for embedding_type in embedding_types
            ]

    repository = Repository()
    service = BrowserImportService(
        settings=Settings(
            import_root=tmp_path,
            understanding_concurrency=1,
            asset_enrichment_queue_size=1,
        ),
        repository=repository,  # type: ignore[arg-type]
        runner=Runner(),  # type: ignore[arg-type]
        understanding_service=Understanding(),  # type: ignore[arg-type]
        embedding_service=Embedding(),  # type: ignore[arg-type]
    )

    result = await service.execute(
        completion=ImportCompletion(
            job_id="job_stream",
            staged_path=tmp_path,
            file_count=1,
        ),
        workspace_id="workspace_stream",
    )

    assert result is not None
    assert events.index("understanding_started") < events.index("runner_finished")
    assert events.index("text_embedding_started") < events.index("job_finalized")
    assert repository.final_asset_ids == ["asset_a"]
    assert repository.failures == []


def test_asset_understanding_normalizes_loose_model_feature_json() -> None:
    understanding = AssetUnderstanding.model_validate(
        {
            "asset_name": "黄昏街景",
            "asset_description": "暖色夕阳下的城市街景。",
            "features": {
                "subject_content": {
                    "value": "城市与行人",
                    "status": "observed",
                    "confidence": 0.9,
                    "evidence": "画面中可直接观察",
                },
                "asset_usage": "动画场景参考",
            },
        }
    )

    assert understanding.features.subject_content.evidence == ["画面中可直接观察"]
    assert understanding.features.asset_usage.value == "动画场景参考"
    assert understanding.features.target_audience.status.value == "unknown"

    inapplicable = AssetUnderstanding.model_validate(
        {
            "asset_name": "空场景",
            "asset_description": "画面中没有人物。",
            "features": {
                "character_state_or_psychology": {
                    "value": "静止",
                    "status": "not_applicable",
                    "confidence": 1.0,
                    "evidence": [],
                }
            },
        }
    )
    assert inapplicable.features.character_state_or_psychology.value is None

    normalized_terms = AssetUnderstanding.model_validate(
        {
            "asset_name": "黄昏街景",
            "asset_description": "暖色夕阳下的城市街景。",
            "features": {
                "visual_style": {
                    "value": ["数字插画", "赛博艺术", "数字插画"],
                    "status": "observed",
                    "confidence": 0.9,
                    "evidence": [],
                }
            },
        }
    )
    assert normalized_terms.features.visual_style.value == "数字插画；赛博艺术"

    with pytest.raises(ValidationError, match="features must be an object"):
        AssetUnderstanding.model_validate(
            {
                "asset_name": "黄昏街景",
                "asset_description": "暖色夕阳下的城市街景。",
                "features": [
                    {
                        "feature_name": "scene_theme",
                        "value": "都市黄昏",
                        "status": "observed",
                        "confidence": 0.8,
                        "evidence": ["夕阳与建筑"],
                    }
                ],
            }
        )
