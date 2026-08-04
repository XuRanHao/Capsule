import asyncio
from pathlib import Path
from types import SimpleNamespace

from capsule.config import Settings
from capsule.enums import AssetType
from capsule.model_clients.mobileclip import ResidentMobileClipWorker
from capsule.parsers.assetizer import AssetizationResult
from capsule.parsers.discovery import discover_files
from capsule.pipeline import runner as runner_module
from capsule.pipeline.runner import (
    PipelineRunner,
    _collect_image_source_contexts,
    _processing_fingerprint,
)
from capsule.schemas import AssetDraft, DiscoveredFile


def test_build_plan_counts_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Capsule", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Capsule plain text", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")
    (tmp_path / "ignore.pdf").write_bytes(b"pdf")

    plan = PipelineRunner().build_plan(tmp_path, "workspace_demo")

    assert plan.file_count == 4
    assert plan.counts_by_extension == {".md": 1, ".pdf": 1, ".png": 1, ".txt": 1}
    assert plan.workspace_id == "workspace_demo"


def test_video_processing_fingerprint_tracks_adaptive_segmentation_settings() -> None:
    source = DiscoveredFile(
        path="/tmp/demo.mp4",
        relative_path="demo.mp4",
        extension=".mp4",
        size_bytes=1,
    )

    baseline = _processing_fingerprint(source, Settings())
    changed = _processing_fingerprint(
        source,
        Settings(video_max_merge_cost=0.75),
    )

    assert baseline != changed


async def test_runner_reuses_resident_video_worker_across_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeDatabase:
        pass

    class FakeRepository:
        async def create_job(self, **_values: object) -> str:
            return "job_empty"

        async def add_job_stage_durations(self, **_values: object) -> None:
            return None

        async def finalize_job(self, **_values: object) -> None:
            return None

    captured_embedders: list[object] = []

    def capture_assetizer(_counter, _settings, video_embedder):
        captured_embedders.append(video_embedder)
        return object()

    monkeypatch.setattr(runner_module, "AssetRepository", lambda _database: FakeRepository())
    monkeypatch.setattr(runner_module, "_build_assetizer", capture_assetizer)
    runner = PipelineRunner(
        settings=Settings(),
        database=FakeDatabase(),  # type: ignore[arg-type]
    )

    await runner.run(tmp_path, "workspace_first")
    await runner.run(tmp_path, "workspace_second")

    assert len(captured_embedders) == 2
    assert captured_embedders[0] is captured_embedders[1]
    assert isinstance(captured_embedders[0], ResidentMobileClipWorker)


def test_markdown_paragraph_is_attached_to_referenced_image(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "sunset.png").write_bytes(b"image")
    (tmp_path / "board.md").write_text(
        "# 光线参考\n\n午后黄昏呈现金黄色调。\n\n![](images/sunset.png)\n",
        encoding="utf-8",
    )

    contexts = _collect_image_source_contexts(discover_files(tmp_path))

    linked = contexts["images/sunset.png"]
    assert linked[0].text == "午后黄昏呈现金黄色调。"
    assert linked[0].relation_type == "preceding_text"
    assert linked[0].paragraph_id == "board.md#block-1"
    assert linked[0].source_path == "board.md"
    assert linked[0].document_title == "光线参考"
    assert linked[0].heading_path == ["光线参考"]


async def test_run_processes_files_with_bounded_concurrency_and_stable_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for index in range(8):
        (tmp_path / f"file-{index}.png").write_bytes(bytes([index]))

    class FakeDatabase:
        async def dispose(self) -> None:
            raise AssertionError("an injected database must not be disposed")

    class FakeRepository:
        def __init__(self) -> None:
            self.successes = 0
            self.failures = 0
            self.finalized = False

        async def create_job(self, **_values: object) -> str:
            return "job_test"

        async def prepare_source_file(self, **values: object) -> SimpleNamespace:
            return SimpleNamespace(
                source_file_id=f"source_{values['source_file'].relative_path}",
                already_processed=False,
                asset_count=0,
            )

        async def replace_assets(self, **_values: object) -> SimpleNamespace:
            return SimpleNamespace(asset_ids=[])

        async def record_file_success(self, **_values: object) -> None:
            self.successes += 1

        async def record_file_failure(self, **_values: object) -> None:
            self.failures += 1

        async def add_job_stage_durations(self, **_values: object) -> None:
            return None

        async def finalize_job(self, **_values: object) -> None:
            self.finalized = True

    class TrackingAssetizer:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def assetize(self, source_file) -> AssetizationResult:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                index = int(Path(source_file.path).stem.removeprefix("file-"))
                await asyncio.sleep((8 - index) * 0.002)
                if index in {1, 6}:
                    return AssetizationResult(
                        source_file=source_file,
                        succeeded=False,
                        error_message=f"failure {index}",
                    )
                return AssetizationResult(source_file=source_file, succeeded=True)
            finally:
                self.active -= 1

    repository = FakeRepository()
    assetizer = TrackingAssetizer()
    monkeypatch.setattr(runner_module, "AssetRepository", lambda _database: repository)
    monkeypatch.setattr(runner_module, "_build_assetizer", lambda *_args: assetizer)
    runner = PipelineRunner(
        settings=Settings(file_parse_concurrency=3),
        database=FakeDatabase(),  # type: ignore[arg-type]
    )

    result = await runner.run(tmp_path, "workspace_test")

    assert assetizer.max_active == 3
    assert result.succeeded_count == 6
    assert result.failed_count == 2
    assert [error["relative_path"] for error in result.errors] == [
        "file-1.png",
        "file-6.png",
    ]
    assert repository.successes == 6
    assert repository.failures == 2
    assert repository.finalized


async def test_run_skips_completed_source_with_same_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "unchanged.png"
    source.write_bytes(b"unchanged")

    class FakeDatabase:
        pass

    class FakeRepository:
        async def create_job(self, **_values: object) -> str:
            return "job_cached"

        async def prepare_source_file(self, **_values: object) -> SimpleNamespace:
            return SimpleNamespace(
                source_file_id="source_cached",
                already_processed=True,
                asset_count=3,
            )

        async def record_file_success(self, **_values: object) -> None:
            return None

        async def add_job_stage_durations(self, **_values: object) -> None:
            return None

        async def finalize_job(self, **_values: object) -> None:
            return None

    class UnexpectedAssetizer:
        async def assetize(self, _source_file) -> AssetizationResult:
            raise AssertionError("a completed identical source must not be parsed again")

    repository = FakeRepository()
    monkeypatch.setattr(runner_module, "AssetRepository", lambda _database: repository)
    monkeypatch.setattr(
        runner_module,
        "_build_assetizer",
        lambda *_args: UnexpectedAssetizer(),
    )
    runner = PipelineRunner(
        settings=Settings(file_parse_concurrency=1),
        database=FakeDatabase(),  # type: ignore[arg-type]
    )

    result = await runner.run(tmp_path, "workspace_test")

    assert result.succeeded_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 1
    assert result.asset_count == 3


async def test_run_emits_committed_assets_and_defers_job_finalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "asset.png").write_bytes(b"image")

    class FakeDatabase:
        pass

    class FakeRepository:
        def __init__(self) -> None:
            self.finalized = False

        async def create_job(self, **_values: object) -> str:
            return "job_stream"

        async def prepare_source_file(self, **_values: object) -> SimpleNamespace:
            return SimpleNamespace(
                source_file_id="source_stream",
                already_processed=False,
                asset_count=0,
            )

        async def replace_assets(self, **_values: object) -> SimpleNamespace:
            return SimpleNamespace(
                asset_ids=["asset_parent", "asset_stream"],
                indexable_asset_ids=["asset_stream"],
            )

        async def record_file_success(self, **_values: object) -> None:
            return None

        async def record_file_failure(self, **_values: object) -> None:
            raise AssertionError("the source file should succeed")

        async def add_job_stage_durations(self, **_values: object) -> None:
            return None

        async def finalize_job(self, **_values: object) -> None:
            self.finalized = True

    class SuccessfulAssetizer:
        async def assetize(self, source_file) -> AssetizationResult:
            return AssetizationResult(source_file=source_file, succeeded=True)

    repository = FakeRepository()
    monkeypatch.setattr(runner_module, "AssetRepository", lambda _database: repository)
    monkeypatch.setattr(
        runner_module,
        "_build_assetizer",
        lambda *_args: SuccessfulAssetizer(),
    )
    committed: list[str] = []

    async def on_assets_stored(asset_ids: list[str]) -> None:
        committed.extend(asset_ids)

    runner = PipelineRunner(
        settings=Settings(file_parse_concurrency=1),
        database=FakeDatabase(),  # type: ignore[arg-type]
    )
    result = await runner.run(
        tmp_path,
        "workspace_stream",
        on_assets_stored=on_assets_stored,
        finalize_job=False,
    )

    assert result.asset_ids == ["asset_parent", "asset_stream"]
    assert result.indexable_asset_ids == ["asset_stream"]
    assert committed == ["asset_stream"]
    assert not repository.finalized


async def test_video_segments_are_committed_and_emitted_one_by_one(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    events: list[str] = []

    class FakeDatabase:
        pass

    class FakeRepository:
        def __init__(self) -> None:
            self.asset_ids: list[str] = []

        async def create_job(self, **_values: object) -> str:
            return "job_video"

        async def prepare_source_file(self, **_values: object) -> SimpleNamespace:
            return SimpleNamespace(
                source_file_id="source_video",
                already_processed=False,
                asset_count=0,
                generation=7,
            )

        async def assert_current_generation(self, **_values: object) -> None:
            return None

        async def upsert_generated_asset(self, *, asset, **_values: object) -> str:
            events.append(f"upsert:{asset.source_locator['start_ms']}")
            self.asset_ids.append(asset.asset_id)
            return asset.asset_id

        async def finalize_asset_generation_if_complete(self, **_values: object) -> bool:
            events.append("generation-check")
            return len(self.asset_ids) == 2

        async def finalize_asset_generation(self, **_values: object) -> SimpleNamespace:
            events.append("source-finalize")
            return SimpleNamespace(asset_ids=list(self.asset_ids))

        async def replace_assets(self, **_values: object) -> SimpleNamespace:
            raise AssertionError("video Assets must not use whole-file replace")

        async def record_file_success(self, **_values: object) -> None:
            return None

        async def record_file_failure(self, **_values: object) -> None:
            raise AssertionError("the video source should succeed")

        async def add_job_stage_durations(self, **_values: object) -> None:
            return None

        async def finalize_job(self, **_values: object) -> None:
            return None

    class VideoAssetizer:
        async def assetize(self, source_file) -> AssetizationResult:
            return AssetizationResult(
                source_file=source_file,
                succeeded=True,
                assets=[
                    AssetDraft(
                        asset_type=AssetType.VIDEO_SEGMENT,
                        file_name=source.name,
                        source_locator={"start_ms": index * 1_000, "end_ms": (index + 1) * 1_000},
                        file_info={
                            "fps": 30.0,
                            "representative_frames": [{"timestamp_ms": index * 1_000 + 500}],
                        },
                    )
                    for index in range(2)
                ],
            )

    class FakeMediaWriter:
        def __init__(self, _storage, **values: object) -> None:
            self._callback = values["on_asset_persisted"]

        async def persist(self, *, source_file, assets, on_asset_committed=None):
            del source_file
            persisted = []
            for asset in assets:
                updated = asset.model_copy(
                    update={"derived_file_uri": f"s3://bucket/{asset.asset_key}.mp4"}
                )
                asset_id = await self._callback(updated, len(assets))
                if on_asset_committed is not None:
                    await on_asset_committed(asset_id)
                persisted.append(updated)
            return persisted

        async def start(self) -> list[tuple[str, str]]:
            return []

        async def close(self) -> None:
            return None

    repository = FakeRepository()
    monkeypatch.setattr(runner_module, "AssetRepository", lambda _database: repository)
    monkeypatch.setattr(runner_module, "_build_assetizer", lambda *_args: VideoAssetizer())
    monkeypatch.setattr(runner_module, "VideoDerivedMediaWriter", FakeMediaWriter)

    async def on_assets_stored(asset_ids: list[str]) -> None:
        events.append(f"understanding:{asset_ids[0]}")

    result = await PipelineRunner(
        settings=Settings(file_parse_concurrency=1),
        database=FakeDatabase(),  # type: ignore[arg-type]
        object_storage=object(),  # type: ignore[arg-type]
    ).run(tmp_path, "workspace_video", on_assets_stored=on_assets_stored)

    assert result.asset_count == 2
    assert events[-1] == "source-finalize"
    assert [event for event in events if event.startswith("upsert:")] == [
        "upsert:0",
        "upsert:1000",
    ]
    assert len([event for event in events if event.startswith("understanding:")]) == 2
    assert all(event.startswith("understanding:") for event in (events[2], events[5]))
