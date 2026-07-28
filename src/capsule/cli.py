import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config

from capsule import __version__
from capsule.bootstrap import bootstrap_runtime
from capsule.config import get_settings
from capsule.parsers import discover_files
from capsule.parsers.video import VideoParser
from capsule.pipeline.runner import PipelineRunner
from capsule.search.evaluation import evaluate_search_file

app = typer.Typer(no_args_is_help=True, help="Capsule multimodal clustering pipeline")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor(
    require_model: Annotated[
        bool,
        typer.Option(
            "--require-model/--allow-missing-model",
            help="Fail when CAPSULE_ARK_API_KEY is not configured.",
        ),
    ] = False,
) -> None:
    """Inspect local configuration and external binary availability."""
    settings = get_settings()
    checks = {
        "database_url": bool(settings.database_url),
        "milvus_uri": bool(settings.milvus_uri),
        "object_storage_endpoint": bool(settings.object_storage_endpoint),
        "ffmpeg": "ffmpeg" not in VideoParser.check_dependencies(),
        "ffprobe": "ffprobe" not in VideoParser.check_dependencies(),
        "ark_api_key": settings.ark_api_key is not None,
    }
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))
    required_checks = {key: value for key, value in checks.items() if key != "ark_api_key"}
    if require_model:
        required_checks["ark_api_key"] = checks["ark_api_key"]
    if not all(required_checks.values()):
        raise typer.Exit(code=1)


@app.command()
def bootstrap(
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
    workspace_name: Annotated[str, typer.Option("--workspace-name")] = "Capsule Demo",
) -> None:
    """Migrate PostgreSQL and initialize the local workspace, bucket, and Milvus."""
    alembic_config = Config("alembic.ini")
    command.upgrade(alembic_config, "head")
    result = asyncio.run(
        bootstrap_runtime(
            get_settings(),
            workspace_id=workspace,
            workspace_name=workspace_name,
        )
    )
    typer.echo(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


@app.command()
def scan(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
) -> None:
    """List supported files without modifying storage or databases."""
    files = discover_files(input_path)
    typer.echo(
        json.dumps(
            [item.model_dump() for item in files],
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command(name="pipeline")
def pipeline_command(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--execute",
            help="Only build the plan until adapters are implemented.",
        ),
    ] = True,
) -> None:
    """Build or execute the staged asset-processing pipeline."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runner = PipelineRunner()
    if dry_run:
        typer.echo(runner.build_plan(input_path, workspace).model_dump_json(indent=2))
        return

    result = asyncio.run(runner.run(input_path, workspace))
    typer.echo(result.model_dump_json(indent=2))
    if result.failed_count:
        raise typer.Exit(code=2)


@app.command(name="mps-video")
def mps_video_command(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
) -> None:
    """Import one MP4/MOV using the macOS-host MPS MobileCLIP runtime."""
    if input_path.suffix.lower() not in {".mp4", ".mov"}:
        raise typer.BadParameter("mps-video accepts a single .mp4 or .mov file")

    result = asyncio.run(PipelineRunner().run(input_path, workspace))
    typer.echo(result.model_dump_json(indent=2))
    if result.failed_count:
        raise typer.Exit(code=2)


@app.command(name="evaluate-search")
def evaluate_search_command(
    dataset: Annotated[Path, typer.Argument(exists=True, readable=True)],
    api_base_url: Annotated[
        str,
        typer.Option("--api-base-url"),
    ] = "http://localhost:8010",
    concurrency: Annotated[int, typer.Option(min=1, max=32)] = 4,
    strict: Annotated[
        bool,
        typer.Option("--strict/--report-only"),
    ] = False,
) -> None:
    """Measure Precision@5 and Recall@10 from a labeled JSONL dataset."""
    report = asyncio.run(
        evaluate_search_file(
            dataset,
            api_base_url=api_base_url,
            concurrency=concurrency,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if strict and not report.passed:
        raise typer.Exit(code=1)
