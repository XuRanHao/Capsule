import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from alembic import command
from alembic.config import Config

from capsule import __version__
from capsule.bootstrap import bootstrap_runtime
from capsule.config import get_settings
from capsule.db.repositories import (
    AssetRepository,
    ClusterRepository,
    CurrentClusterRepository,
    EmbeddingRepository,
)
from capsule.db.session import Database
from capsule.enums import ClusterAlgorithm, EmbeddingType
from capsule.features import ACTIVE_EMBEDDING_TYPES
from capsule.media.model_image import ModelImageCache
from capsule.model_clients.doubao import DoubaoClient
from capsule.parsers import discover_files
from capsule.parsers.video import VideoParser
from capsule.pipeline.cluster_service import ClusterService, EmbeddingTypeClusterResult
from capsule.pipeline.embedding import AssetEmbeddingService, EmbeddingRunResult
from capsule.pipeline.cloud_source_dispatcher import CloudSourceTaskDispatcher
from capsule.pipeline.import_service import AssetEnrichmentResult, enrich_assets
from capsule.pipeline.processing_task_service import (
    CpuProcessingTaskScheduler,
    CpuProcessingTaskWorker,
    ProcessingTaskSubmissionService,
)
from capsule.pipeline.runner import PipelineRunner
from capsule.pipeline.search_vector_index import SearchVectorMaterializationResult
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.pipeline.video_task_runtime import ProcessingTaskKind
from capsule.pipeline.video_task_service import (
    VideoTaskScheduler,
    VideoTaskSubmissionService,
    VideoTaskWorker,
)
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


@app.command(name="submit-video-task")
def submit_video_task_command(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
) -> None:
    """Create the durable PostgreSQL task, then publish one video delivery."""
    result = asyncio.run(
        VideoTaskSubmissionService().submit(input_path=input_path, workspace_id=workspace)
    )
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))


@app.command(name="media-worker")
@app.command(name="video-worker")
def video_worker_command(
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    once: Annotated[
        bool,
        typer.Option("--once", help="Process one Streams delivery and exit."),
    ] = False,
) -> None:
    """Run the resident-MPS, PostgreSQL-fenced video/audio media worker."""
    worker = VideoTaskWorker.from_settings(worker_id=worker_id)
    if once:
        outcome = asyncio.run(_run_video_worker_once(worker))
        typer.echo(json.dumps({"outcome": outcome}, ensure_ascii=False))
        return
    asyncio.run(worker.run_forever())


@app.command(name="media-scheduler")
@app.command(name="video-scheduler")
def video_scheduler_command(
    once: Annotated[
        bool,
        typer.Option("--once", help="Recover and publish due tasks once, then exit."),
    ] = False,
) -> None:
    """Recover expired leases and dispatch PostgreSQL-backed media retries."""
    scheduler = VideoTaskScheduler.from_settings()
    if once:
        published = asyncio.run(_run_video_scheduler_once(scheduler))
        typer.echo(json.dumps({"published": published}, ensure_ascii=False))
        return
    asyncio.run(scheduler.run_forever())


async def _run_video_worker_once(worker: VideoTaskWorker) -> str:
    try:
        return await worker.run_once()
    finally:
        await worker.close()


async def _run_video_scheduler_once(scheduler: VideoTaskScheduler) -> int:
    try:
        return await scheduler.run_once()
    finally:
        await scheduler.close()


@app.command(name="submit-processing-task")
def submit_processing_task_command(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
) -> None:
    """Create and publish one durable image or text assetization task."""
    result = asyncio.run(
        ProcessingTaskSubmissionService().submit(
            input_path=input_path,
            workspace_id=workspace,
        )
    )
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))


@app.command(name="cpu-task-worker")
def cpu_task_worker_command(
    task_kind: Annotated[ProcessingTaskKind, typer.Option("--kind")],
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    once: Annotated[bool, typer.Option("--once")] = False,
) -> None:
    """Run one trusted image or text CPU task worker pool."""
    if task_kind is ProcessingTaskKind.VIDEO:
        raise typer.BadParameter("use video-worker for video tasks")
    worker = CpuProcessingTaskWorker.for_kind(
        task_kind=task_kind,
        worker_id=worker_id,
    )
    if once:
        outcome = asyncio.run(_run_video_worker_once(worker))
        typer.echo(json.dumps({"outcome": outcome}, ensure_ascii=False))
        return
    asyncio.run(worker.run_forever())


@app.command(name="cpu-task-scheduler")
def cpu_task_scheduler_command(
    task_kind: Annotated[ProcessingTaskKind, typer.Option("--kind")],
    once: Annotated[bool, typer.Option("--once")] = False,
) -> None:
    """Recover and redispatch one image or text CPU task route."""
    if task_kind is ProcessingTaskKind.VIDEO:
        raise typer.BadParameter("use video-scheduler for video tasks")
    scheduler = CpuProcessingTaskScheduler.for_kind(task_kind=task_kind)
    if once:
        published = asyncio.run(_run_video_scheduler_once(scheduler))
        typer.echo(json.dumps({"published": published}, ensure_ascii=False))
        return
    asyncio.run(scheduler.run_forever())


@app.command(name="cloud-source-dispatcher")
def cloud_source_dispatcher_command(
    once: Annotated[
        bool,
        typer.Option("--once", help="Create durable tasks for pending S3/MinIO sources once."),
    ] = False,
) -> None:
    """Turn externally registered ``source_files`` cloud objects into durable tasks."""
    dispatcher = CloudSourceTaskDispatcher()
    if once:
        created = asyncio.run(_run_cloud_source_dispatcher_once(dispatcher))
        typer.echo(json.dumps({"created": created}, ensure_ascii=False))
        return
    asyncio.run(dispatcher.run_forever())


async def _run_cloud_source_dispatcher_once(dispatcher: CloudSourceTaskDispatcher) -> int:
    try:
        return await dispatcher.run_once()
    finally:
        await dispatcher.close()


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
    algorithm: Annotated[
        ClusterAlgorithm,
        typer.Option("--algorithm", help="Clustering algorithm."),
    ] = ClusterAlgorithm.COMPLETE_LINK,
    distance_threshold: Annotated[
        float,
        typer.Option(
            "--distance-threshold",
            min=0.01,
            max=2.0,
            help="Complete-link cutoff on L2-normalized vectors.",
        ),
    ] = 0.5,
    min_cluster_size: Annotated[
        int,
        typer.Option(
            "--min-cluster-size",
            min=2,
            max=10_000,
            help="Minimum retained cluster size; Complete-link defaults to two assets.",
        ),
    ] = 2,
    optimize_parameters: Annotated[
        bool,
        typer.Option(
            "--optimize-parameters/--no-optimize-parameters",
            help="Evaluate multiple HDBSCAN parameter candidates; disabled by default.",
        ),
    ] = False,
) -> None:
    """Cluster one Embedding Type into its own PCA and ClusterRun."""
    if optimize_parameters and algorithm is not ClusterAlgorithm.HDBSCAN:
        raise typer.BadParameter("--optimize-parameters is only available with --algorithm hdbscan")
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    result = asyncio.run(
        _cluster_assets(
            workspace_id=workspace,
            embedding_type=embedding_type,
            algorithm=algorithm,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
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
    force_understanding: Annotated[
        bool,
        typer.Option(
            "--force-understanding",
            help="Regenerate descriptions and Features before embedding.",
        ),
    ] = False,
) -> None:
    """Backfill understanding and all embedding channels for stored Assets."""
    result = asyncio.run(
        _enrich_assets(
            job_id=job_id,
            workspace_id=workspace,
            asset_ids=asset_ids,
            force_understanding=force_understanding,
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


@app.command(name="materialize-search-vectors")
def materialize_search_vectors_command(
    workspace: Annotated[str, typer.Option("--workspace")] = "workspace_demo",
) -> None:
    """Rebuild fixed 0.3/0.7 search vectors from existing raw embeddings."""
    result = asyncio.run(_materialize_search_vectors(workspace_id=workspace))
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, indent=2))


async def _materialize_search_vectors(*, workspace_id: str) -> SearchVectorMaterializationResult:
    settings = get_settings()
    database = Database(settings)
    model_client = DoubaoClient(settings)
    try:
        service = AssetEmbeddingService(
            settings=settings,
            repository=EmbeddingRepository(database),
            model_client=model_client,
            vector_store=MilvusVectorStore(settings),
            artifact_reader=ObjectStorage(settings),
        )
        return await service.materialize_search_vectors(
            workspace_id=workspace_id,
            embedding_types=[
                embedding_type
                for embedding_type in ACTIVE_EMBEDDING_TYPES
                if embedding_type is not EmbeddingType.NATIVE_MULTIMODAL
            ],
        )
    finally:
        await model_client.close()
        await database.dispose()


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
                artifact_reader=ObjectStorage(settings),
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
    algorithm: ClusterAlgorithm = ClusterAlgorithm.COMPLETE_LINK,
    distance_threshold: float = 0.5,
    min_cluster_size: int = 2,
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
                current_cluster_repository=CurrentClusterRepository(database),
                vector_store=MilvusVectorStore(settings),
                model_client=model_client,
            )
            return await service.run(
                workspace_id=workspace_id,
                embedding_type=embedding_type,
                algorithm=algorithm,
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                optimize_parameters=optimize_parameters,
            )
    finally:
        await database.dispose()


async def _enrich_assets(
    *,
    job_id: str,
    workspace_id: str,
    asset_ids: list[str] | None,
    force_understanding: bool = False,
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
            understanding_image_cache = ModelImageCache(
                target_bytes=settings.model_image_target_bytes,
                max_edge=settings.model_image_max_edge,
                fixed_size=settings.understanding_image_size,
                max_entries=settings.model_image_cache_entries,
            )
            embedding_image_cache = ModelImageCache(
                target_bytes=settings.model_image_target_bytes,
                max_edge=settings.model_image_max_edge,
                max_entries=settings.model_image_cache_entries,
            )
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
                    image_cache=understanding_image_cache,
                ),
                embedding_service=AssetEmbeddingService(
                    settings=settings,
                    repository=embedding_repository,
                    model_client=model_client,
                    vector_store=MilvusVectorStore(settings),
                    artifact_reader=storage,
                    image_cache=embedding_image_cache,
                ),
                force_understanding=force_understanding,
            )
    finally:
        await database.dispose()
