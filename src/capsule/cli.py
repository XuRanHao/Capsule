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
from capsule.db.repositories import AssetRepository, ClusterRepository, EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import EmbeddingType
from capsule.model_clients.doubao import DoubaoClient
from capsule.parsers import discover_files
from capsule.parsers.video import VideoParser
from capsule.pipeline.cluster_service import ClusterService, EmbeddingTypeClusterResult
from capsule.pipeline.embedding import AssetEmbeddingService, EmbeddingRunResult
from capsule.pipeline.import_service import AssetEnrichmentResult, enrich_assets
from capsule.pipeline.runner import PipelineRunner
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.search.evaluation import evaluate_search_file
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.milvus import MilvusVectorStore

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


@app.command(name="embed")
def embed_command(
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option("--embedding-type", help="Run exactly one independent embedding channel."),
    ] = EmbeddingType.NATIVE_MULTIMODAL,
    asset_ids: Annotated[
        list[str] | None,
        typer.Option("--asset-id", help="Only embed this Asset ID; repeat the option as needed."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Regenerate and upsert already indexed logical inputs."),
    ] = False,
) -> None:
    """Generate Embeddings for stored Assets and upsert them into Milvus."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = asyncio.run(
        _embed_assets(
            workspace_id=workspace,
            embedding_type=embedding_type,
            asset_ids=asset_ids,
            force=force,
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.failed_count:
        raise typer.Exit(code=2)


@app.command(name="cluster")
def cluster_command(
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
    embedding_type: Annotated[
        EmbeddingType,
        typer.Option(
            "--embedding-type",
            help="Embedding Type to cluster; each invocation runs exactly one Type.",
        ),
    ] = EmbeddingType.NATIVE_MULTIMODAL,
    optimize_parameters: Annotated[
        bool,
        typer.Option(
            "--optimize-parameters/--no-optimize-parameters",
            help="Evaluate multiple HDBSCAN parameter candidates; disabled by default.",
        ),
    ] = False,
) -> None:
    """Cluster one Embedding Type into its own PCA, HDBSCAN, and ClusterRun."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = asyncio.run(
        _cluster_assets(
            workspace_id=workspace,
            embedding_type=embedding_type,
            optimize_parameters=optimize_parameters,
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.status.value == "failed":
        raise typer.Exit(code=2)


@app.command(name="enrich")
def enrich_command(
    job_id: Annotated[str, typer.Option("--job-id", help="Processing Job to update.")],
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
    asset_ids: Annotated[
        list[str] | None,
        typer.Option("--asset-id", help="Only enrich this Asset ID; repeat as needed."),
    ] = None,
) -> None:
    """Backfill understanding and all embedding channels for stored Assets."""
    result = asyncio.run(
        _enrich_assets(
            job_id=job_id,
            workspace_id=workspace,
            asset_ids=asset_ids,
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.partial_failed_asset_count:
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


async def _embed_assets(
    *,
    workspace_id: str,
    embedding_type: EmbeddingType,
    asset_ids: list[str] | None,
    force: bool,
) -> EmbeddingRunResult:
    settings = get_settings()
    database = Database(settings)
    try:
        async with DoubaoClient(settings) as model_client:
            service = AssetEmbeddingService(
                settings=settings,
                repository=EmbeddingRepository(database),
                model_client=model_client,
                vector_store=MilvusVectorStore(settings),
                video_url_signer=ObjectStorage(settings),
            )
            return await service.run(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                asset_ids=asset_ids,
                force=force,
            )
    finally:
        await database.dispose()


async def _cluster_assets(
    *,
    workspace_id: str,
    embedding_type: EmbeddingType,
    optimize_parameters: bool = False,
) -> EmbeddingTypeClusterResult:
    settings = get_settings()
    database = Database(settings)
    try:
        async with DoubaoClient(settings) as model_client:
            service = ClusterService(
                settings=settings,
                embedding_repository=EmbeddingRepository(database),
                cluster_repository=ClusterRepository(database),
                vector_store=MilvusVectorStore(settings),
                model_client=model_client,
            )
            return await service.run(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                optimize_parameters=optimize_parameters,
            )
    finally:
        await database.dispose()


async def _enrich_assets(
    *,
    job_id: str,
    workspace_id: str,
    asset_ids: list[str] | None,
) -> AssetEnrichmentResult:
    settings = get_settings()
    database = Database(settings)
    storage = ObjectStorage(settings)
    asset_repository = AssetRepository(database)
    embedding_repository = EmbeddingRepository(database)
    try:
        assets = await embedding_repository.list_assets(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
        )
        selected_asset_ids = [asset.asset_id for asset in assets]
        if not selected_asset_ids:
            raise ValueError("no Assets matched the enrichment request")
        async with DoubaoClient(settings) as model_client:
            return await enrich_assets(
                job_id=job_id,
                workspace_id=workspace_id,
                asset_ids=selected_asset_ids,
                repository=asset_repository,
                understanding_service=AssetUnderstandingService(
                    settings=settings,
                    embedding_repository=embedding_repository,
                    asset_repository=asset_repository,
                    model_client=model_client,
                    artifact_reader=storage,
                ),
                embedding_service=AssetEmbeddingService(
                    settings=settings,
                    repository=embedding_repository,
                    model_client=model_client,
                    vector_store=MilvusVectorStore(settings),
                    video_url_signer=storage,
                ),
            )
    finally:
        await database.dispose()
