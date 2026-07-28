import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from capsule import __version__
from capsule.config import get_settings
from capsule.parsers import discover_files
from capsule.parsers.video import VideoParser
from capsule.pipeline.runner import PipelineNotReadyError, PipelineRunner

app = typer.Typer(no_args_is_help=True, help="Capsule multimodal clustering pipeline")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor() -> None:
    """Inspect local configuration and external binary availability."""
    settings = get_settings()
    checks = {
        "database_url": bool(settings.database_url),
        "milvus_uri": bool(settings.milvus_uri),
        "object_storage_endpoint": bool(settings.object_storage_endpoint),
        "ark_api_key": settings.ark_api_key is not None,
        "ffmpeg": "ffmpeg" not in VideoParser.check_dependencies(),
        "ffprobe": "ffprobe" not in VideoParser.check_dependencies(),
    }
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise typer.Exit(code=1)


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

    try:
        asyncio.run(runner.run(input_path, workspace))
    except PipelineNotReadyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
