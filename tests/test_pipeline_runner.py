from pathlib import Path

from capsule.pipeline.runner import PipelineRunner


def test_build_plan_counts_supported_files(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Capsule", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"image")
    (tmp_path / "ignore.pdf").write_bytes(b"pdf")

    plan = PipelineRunner().build_plan(tmp_path, "workspace_demo")

    assert plan.file_count == 2
    assert plan.counts_by_extension == {".md": 1, ".png": 1}
    assert plan.workspace_id == "workspace_demo"
