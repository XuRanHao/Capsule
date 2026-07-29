import asyncio
from pathlib import Path
from types import SimpleNamespace

from capsule.config import Settings
from capsule.parsers.assetizer import AssetizationResult
from capsule.pipeline import runner as runner_module
from capsule.pipeline.runner import PipelineRunner


def test_build_plan_counts_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Capsule", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Capsule plain text", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")
    (tmp_path / "ignore.pdf").write_bytes(b"pdf")

    plan = PipelineRunner().build_plan(tmp_path, "workspace_demo")

    assert plan.file_count == 3
    assert plan.counts_by_extension == {".md": 1, ".png": 1, ".txt": 1}
    assert plan.workspace_id == "workspace_demo"


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
