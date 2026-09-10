import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from capsule.config import Settings
from capsule.enums import EmbeddingType, PipelineStage
from capsule.features import ACTIVE_EMBEDDING_TYPES
from capsule.pipeline.import_service import (
    BrowserImportService,
    ImportCompletion,
    ImportWorkflowCoordinator,
    enrich_assets,
)
from capsule.schemas import AssetUnderstanding, ProcessingJobRecord


class FakeAssetRepository:
    def __init__(self) -> None:
        self.import_root: Path | None = None
        self.status = "queued"
        self.started_count: int | None = None
        self.upload_activity_count = 0

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

    async def start_import_job(
        self,
        *,
        job_id: str,
        total_count: int,
        post_asset_action: str = "none",
    ) -> None:
        assert job_id == "job_import_test"
        assert post_asset_action in {"none", "enrich"}
        self.status = "running"
        self.started_count = total_count

    async def mark_import_upload_activity(self, *, job_id: str, workspace_id: str) -> None:
        assert job_id == "job_import_test"
        assert workspace_id == "workspace_import_test"
        self.upload_activity_count += 1

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
    assert repository.upload_activity_count == 2
    assert runner.calls == [
        {
            "input_path": job.staged_path,
            "workspace_id": "workspace_import_test",
            "job_id": "job_import_test",
        }
    ]


@pytest.mark.asyncio
async def test_browser_import_dispatches_one_existing_job_to_durable_tasks(
    tmp_path: Path,
) -> None:
    repository = FakeAssetRepository()
    runner = FakeRunner()

    class DurableSubmitter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def submit_batch(self, **values: object) -> list[SimpleNamespace]:
            self.calls.append(values)
            source_files = values["source_files"]
            assert isinstance(source_files, list)
            return [
                SimpleNamespace(task_id=f"task-{index}")
                for index, _ in enumerate(source_files)
            ]

    durable = DurableSubmitter()
    service = BrowserImportService(
        settings=Settings(import_root=tmp_path / "imports"),
        repository=repository,  # type: ignore[arg-type]
        runner=runner,  # type: ignore[arg-type]
        durable_task_submitter=durable,  # type: ignore[arg-type]
    )
    job = await service.create_job(workspace_id="workspace_import_test")
    await service.upload_file(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
        file=UploadFile(filename="note.md", file=BytesIO(b"# Durable")),
        relative_path="notes/note.md",
    )
    completion = await service.complete_job(
        job_id=job.job_id,
        workspace_id="workspace_import_test",
    )

    result = await service.execute(
        completion=completion,
        workspace_id="workspace_import_test",
    )

    assert result is not None
    assert result.file_count == 1
    assert completion.durable_dispatched
    assert repository.started_count is None
    assert runner.calls == []
    assert len(durable.calls) == 1
    assert durable.calls[0]["job_id"] == "job_import_test"
    assert durable.calls[0]["post_asset_action"] == "none"


@pytest.mark.asyncio
async def test_browser_import_can_cancel_an_active_execution(tmp_path: Path) -> None:
    repository = FakeAssetRepository()
    started = asyncio.Event()

    class BlockingRunner:
        async def run(self, *_: object, **__: object) -> SimpleNamespace:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled execution must not resume")

    service = BrowserImportService(
        settings=Settings(import_root=tmp_path / "imports"),
        repository=repository,  # type: ignore[arg-type]
        runner=BlockingRunner(),  # type: ignore[arg-type]
    )
    execution = asyncio.create_task(
        service.execute(
            completion=ImportCompletion(
                job_id="job_import_test",
                staged_path=tmp_path,
                file_count=1,
            ),
            workspace_id="workspace_import_test",
        )
    )
    await started.wait()

    cancelled_count = await service.cancel_active_jobs(
        workspace_id="workspace_import_test"
    )

    assert cancelled_count == 1
    assert await execution is None


@pytest.mark.asyncio
async def test_import_workflow_coordinator_resumes_ready_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Repository:
        async def claim_ready_import_workflow(self, **_: object) -> tuple[str, str, str]:
            return "job-1", "workspace-1", "lease-1"

        async def list_import_job_asset_ids(self, *, job_id: str) -> list[str]:
            assert job_id == "job-1"
            return ["asset-1"]

        async def release_import_workflow(self, **_: object) -> bool:
            raise AssertionError("successful enrichment must not release its lease")

        async def heartbeat_import_workflow(self, **_: object) -> bool:
            return True

    async def fake_enrich_assets(**values: object) -> SimpleNamespace:
        calls.append(values)
        return SimpleNamespace()

    monkeypatch.setattr(
        "capsule.pipeline.import_service.enrich_assets",
        fake_enrich_assets,
    )
    coordinator = ImportWorkflowCoordinator(
        repository=Repository(),  # type: ignore[arg-type]
        understanding_service=object(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="workflow-worker",
    )

    assert await coordinator.run_once()
    assert len(calls) == 1
    assert calls[0]["job_id"] == "job-1"
    assert calls[0]["asset_ids"] == ["asset-1"]
    assert calls[0]["workflow_lease_token"] == "lease-1"


@pytest.mark.asyncio
async def test_import_workflow_coordinator_cancels_enrichment_when_lease_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    enrichment_started = asyncio.Event()
    enrichment_cancelled = asyncio.Event()

    class Repository:
        heartbeat_calls = 0

        async def claim_ready_import_workflow(self, **_: object) -> tuple[str, str, str]:
            return "job-1", "workspace-1", "lease-1"

        async def list_import_job_asset_ids(self, *, job_id: str) -> list[str]:
            assert job_id == "job-1"
            return ["asset-1"]

        async def heartbeat_import_workflow(self, **_: object) -> bool:
            self.heartbeat_calls += 1
            if self.heartbeat_calls == 1:
                return True
            await enrichment_started.wait()
            events.append("lease-lost")
            return False

        async def release_import_workflow(self, **_: object) -> bool:
            events.append("released")
            return True

    async def fake_enrich_assets(**_: object) -> SimpleNamespace:
        events.append("enrichment-started")
        enrichment_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("enrichment-cancelled")
            enrichment_cancelled.set()
            raise
        events.extend(("stage", "finalize"))
        return SimpleNamespace()

    monkeypatch.setattr(
        "capsule.pipeline.import_service.enrich_assets",
        fake_enrich_assets,
    )
    coordinator = ImportWorkflowCoordinator(
        repository=Repository(),  # type: ignore[arg-type]
        understanding_service=object(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="workflow-worker",
        lease_seconds=0.003,
    )

    assert not await coordinator.run_once()
    assert enrichment_cancelled.is_set()
    assert "lease-lost" in events
    assert "stage" not in events
    assert "finalize" not in events
    assert events[-1] == "released"


@pytest.mark.asyncio
async def test_import_workflow_coordinator_retries_after_claim_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enriched = asyncio.Event()

    class Repository:
        claim_calls = 0

        async def claim_ready_import_workflow(self, **_: object) -> tuple[str, str, str] | None:
            self.claim_calls += 1
            if self.claim_calls == 1:
                raise RuntimeError("temporary database failure")
            if self.claim_calls == 2:
                return "job-1", "workspace-1", "lease-1"
            return None

        async def list_import_job_asset_ids(self, *, job_id: str) -> list[str]:
            assert job_id == "job-1"
            return ["asset-1"]

        async def heartbeat_import_workflow(self, **_: object) -> bool:
            return True

        async def release_import_workflow(self, **_: object) -> bool:
            raise AssertionError("successful enrichment must not release its lease")

    async def fake_enrich_assets(**_: object) -> SimpleNamespace:
        enriched.set()
        return SimpleNamespace()

    monkeypatch.setattr(
        "capsule.pipeline.import_service.enrich_assets",
        fake_enrich_assets,
    )
    coordinator = ImportWorkflowCoordinator(
        repository=Repository(),  # type: ignore[arg-type]
        understanding_service=object(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="workflow-worker",
        poll_seconds=0.001,
    )
    runner = asyncio.create_task(coordinator.run_forever())

    await asyncio.wait_for(enriched.wait(), timeout=1)
    await coordinator.close()
    await asyncio.wait_for(runner, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["run_once", "run_forever"])
async def test_import_workflow_coordinator_cancellation_cleans_up_enrichment_and_lease(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    events: list[str] = []
    enrichment_started = asyncio.Event()
    enrichment_cancelled = asyncio.Event()

    class Repository:
        async def claim_ready_import_workflow(self, **_: object) -> tuple[str, str, str]:
            return "job-1", "workspace-1", "lease-1"

        async def list_import_job_asset_ids(self, *, job_id: str) -> list[str]:
            assert job_id == "job-1"
            return ["asset-1"]

        async def heartbeat_import_workflow(self, **_: object) -> bool:
            return True

        async def release_import_workflow(self, **values: object) -> bool:
            assert values["error"] == "import workflow coordinator cancelled"
            events.append("lease-released")
            return True

    async def fake_enrich_assets(**_: object) -> SimpleNamespace:
        enrichment_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("enrichment-cancelled")
            enrichment_cancelled.set()
            raise
        raise AssertionError("cancelled enrichment must not resume")

    monkeypatch.setattr(
        "capsule.pipeline.import_service.enrich_assets",
        fake_enrich_assets,
    )
    coordinator = ImportWorkflowCoordinator(
        repository=Repository(),  # type: ignore[arg-type]
        understanding_service=object(),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="workflow-worker",
    )
    task = asyncio.create_task(
        coordinator.run_once()
        if entrypoint == "run_once"
        else coordinator.run_forever()
    )

    await asyncio.wait_for(enrichment_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert enrichment_cancelled.is_set()
    assert events == ["enrichment-cancelled", "lease-released"]


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
            self.materialize_calls: list[dict[str, object]] = []

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

        async def materialize_search_vectors(self, **values: object) -> SimpleNamespace:
            self.materialize_calls.append(values)
            raise AssertionError("import enrichment must not materialize search vectors")

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
    assert embedding.types == list(ACTIVE_EMBEDDING_TYPES)
    assert embedding.materialize_calls == []
    assert result.completed_asset_count == 1
    assert result.partial_failed_asset_count == 1
    assert repository.final_errors == [
        {
            "asset_id": "asset_b",
            "stage": "understanding",
            "error": "understanding failed",
        }
    ]
    assert repository.durations["understanding"] == 120.0
    assert repository.durations["feature_ready"] == 5.0
    assert repository.durations["embedding"] > 0
    assert repository.durations["indexing"] > 0
    assert repository.durations["embedding"] / repository.durations[
        "indexing"
    ] == pytest.approx(10)


@pytest.mark.asyncio
async def test_raw_embedding_success_does_not_invoke_search_vector_materialization() -> None:
    class Repository:
        def __init__(self) -> None:
            self.final_errors: list[dict[str, str]] = []

        async def begin_asset_enrichment(self, **_: object) -> None:
            return None

        async def set_job_stage(self, **_: object) -> None:
            return None

        async def add_job_stage_durations(self, **_: object) -> None:
            return None

        async def finalize_enrichment(
            self,
            *,
            errors: list[dict[str, str]],
            **_: object,
        ) -> None:
            self.final_errors = errors

    class Understanding:
        async def run(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                errors=[],
                understanding_duration_ms=1.0,
                feature_ready_duration_ms=1.0,
            )

    class Embedding:
        def __init__(self) -> None:
            self.raw_completed = False
            self.materialize_called = False

        async def run(
            self,
            *,
            embedding_type: EmbeddingType,
            **_: object,
        ) -> SimpleNamespace:
            assert embedding_type is EmbeddingType.NATIVE_MULTIMODAL
            self.raw_completed = True
            return SimpleNamespace(
                embedding_type=embedding_type.value,
                errors=[],
                embedding_duration_ms=1.0,
                indexing_duration_ms=1.0,
            )

        async def run_many(
            self,
            *,
            embedding_types: list[EmbeddingType],
            **_: object,
        ) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    embedding_type=embedding_type.value,
                    errors=[],
                    embedding_duration_ms=1.0,
                    indexing_duration_ms=1.0,
                )
                for embedding_type in embedding_types
            ]

        async def materialize_search_vectors(self, **_: object) -> SimpleNamespace:
            self.materialize_called = True
            raise AssertionError("import enrichment must not materialize search vectors")

    repository = Repository()
    embedding = Embedding()
    result = await enrich_assets(
        job_id="job_test",
        workspace_id="workspace_test",
        asset_ids=["asset_a"],
        repository=repository,  # type: ignore[arg-type]
        understanding_service=Understanding(),  # type: ignore[arg-type]
        embedding_service=embedding,  # type: ignore[arg-type]
    )

    assert embedding.raw_completed is True
    assert embedding.materialize_called is False
    assert result.completed_asset_count == 1
    assert result.partial_failed_asset_count == 0
    assert repository.final_errors == []


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

        async def materialize_search_vectors(self, **_: object) -> SimpleNamespace:
            events.append("search_vectors_materialized")
            raise AssertionError("import enrichment must not materialize search vectors")

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
    assert "search_vectors_materialized" not in events
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
                "scene_theme": "动画场景参考",
            },
        }
    )

    assert understanding.features.subject_content.items[0].evidence == ["画面中可直接观察"]
    assert understanding.features.scene_theme.value == "动画场景参考"
    assert understanding.features.visual_presentation.applicability.value == "unknown"

    inapplicable = AssetUnderstanding.model_validate(
        {
            "asset_name": "空场景",
            "asset_description": "画面中没有人物。",
            "features": {
                "scene_theme": {
                    "value": "静止",
                    "status": "not_applicable",
                    "confidence": 1.0,
                    "evidence": [],
                }
            },
        }
    )
    assert inapplicable.features.scene_theme.value is None

    normalized_terms = AssetUnderstanding.model_validate(
        {
            "asset_name": "黄昏街景",
            "asset_description": "暖色夕阳下的城市街景。",
            "features": {
                "visual_presentation": {
                    "value": ["数字插画", "赛博艺术", "数字插画"],
                    "status": "observed",
                    "confidence": 0.9,
                    "evidence": [],
                }
            },
        }
    )
    assert normalized_terms.features.visual_presentation.value == "数字插画；赛博艺术"

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
