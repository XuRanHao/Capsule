import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from capsule.api.assets import router as assets_router
from capsule.api.capsules import router as capsules_router
from capsule.api.clusters import router as cluster_runs_router
from capsule.api.imports import router as imports_router
from capsule.api.relation_graphs import router as relation_graphs_router
from capsule.api.search import router as search_router
from capsule.api.workspaces import router as workspaces_router
from capsule.config import Settings, get_settings
from capsule.db.repositories import (
    AssetRepository,
    ClusterRepository,
    CurrentClusterRepository,
    EmbeddingRepository,
    RelationGraphRepository,
)
from capsule.db.session import Database
from capsule.media.model_image import ModelImageCache
from capsule.media.video_frames import FFmpegVideoFrameExtractor
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.cluster_service import ClusterService
from capsule.pipeline.durable_task_supervisor import DurableTaskRuntimeSupervisor
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.import_service import BrowserImportService, ImportWorkflowCoordinator
from capsule.pipeline.incremental_clustering import (
    IncrementalAssignmentThresholds,
    IncrementalClusterCoordinator,
    IncrementalClusterService,
)
from capsule.pipeline.processing_task_service import BrowserProcessingTaskSubmissionService
from capsule.pipeline.relation_graph_service import RelationGraphService
from capsule.pipeline.runner import PipelineRunner
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.pipeline.workspace_clear import LibraryClearService
from capsule.pipeline.workspace_management import WorkspaceService
from capsule.search.history import SearchHistoryRepository
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.search.service import SearchService
from capsule.search.uploads import QueryImageService
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.milvus import MilvusVectorStore


def create_app(
    *,
    settings: Settings | None = None,
    search_service: SearchService | None = None,
    cluster_service: ClusterService | None = None,
    cluster_repository: ClusterRepository | None = None,
    current_cluster_repository: CurrentClusterRepository | None = None,
    import_service: BrowserImportService | None = None,
    asset_repository: AssetRepository | None = None,
    library_clear_service: LibraryClearService | None = None,
    workspace_service: WorkspaceService | None = None,
    relation_graph_service: RelationGraphService | None = None,
    entity_hierarchy_checkpointer: Any | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.video_transcode_semaphore = asyncio.Semaphore(
            resolved_settings.video_transcode_concurrency
        )
        if (
            search_service is not None
            or cluster_service is not None
            or cluster_repository is not None
            or current_cluster_repository is not None
            or import_service is not None
            or asset_repository is not None
            or library_clear_service is not None
            or workspace_service is not None
            or relation_graph_service is not None
        ):
            app.state.search_service = search_service
            app.state.cluster_service = cluster_service
            app.state.cluster_repository = cluster_repository
            if current_cluster_repository is not None:
                app.state.current_cluster_repository = current_cluster_repository
            app.state.import_service = import_service
            app.state.asset_repository = asset_repository
            app.state.library_clear_service = library_clear_service
            app.state.workspace_service = workspace_service
            app.state.relation_graph_service = relation_graph_service
            app.state.entity_hierarchy_checkpointer = entity_hierarchy_checkpointer
            yield
            return

        database = Database(resolved_settings)
        storage = ObjectStorage(resolved_settings)
        await storage.ensure_bucket()
        history = SearchHistoryRepository(database, resolved_settings)
        query_images = QueryImageService(database, storage)
        app.state.search_history = history
        app.state.query_image_service = query_images
        app.state.object_storage = storage
        cluster_repo = ClusterRepository(database)
        current_cluster_repo = CurrentClusterRepository(database)
        asset_repo = AssetRepository(database)
        embedding_repository = EmbeddingRepository(database)
        relation_graph_repository = RelationGraphRepository(database)
        vectors = MilvusVectorStore(resolved_settings)
        pipeline_runner = PipelineRunner(
            settings=resolved_settings,
            database=database,
            object_storage=storage,
        )
        durable_task_submitter = BrowserProcessingTaskSubmissionService(
            settings=resolved_settings,
            database=database,
            source_repository=asset_repo,
        )
        durable_task_supervisor: DurableTaskRuntimeSupervisor | None = None
        app.state.durable_task_runtime_mode = "external"
        app.state.durable_task_supervisor = None
        if resolved_settings.api_embedded_cpu_tasks_enabled:
            durable_task_supervisor = DurableTaskRuntimeSupervisor.from_settings(
                settings=resolved_settings
            )
            try:
                await asyncio.wait_for(
                    durable_task_supervisor.start(),
                    timeout=(resolved_settings.api_embedded_cpu_tasks_startup_timeout_seconds),
                )
            except BaseException:
                await durable_task_supervisor.close()
                raise
            app.state.durable_task_runtime_mode = "embedded"
            app.state.durable_task_supervisor = durable_task_supervisor
        app.state.cluster_repository = cluster_repo
        app.state.current_cluster_repository = current_cluster_repo
        app.state.asset_repository = asset_repo
        app.state.library_clear_service = LibraryClearService(
            settings=resolved_settings,
            repository=asset_repo,
            vector_store=vectors,
            object_storage=storage,
        )
        app.state.workspace_service = WorkspaceService(
            settings=resolved_settings,
            repository=asset_repo,
            vector_store=vectors,
            object_storage=storage,
        )
        if resolved_settings.ark_api_key is None:
            logging.getLogger(__name__).warning(
                "CAPSULE_ARK_API_KEY is not configured; search endpoint will return 503"
            )
            app.state.search_service = None
            app.state.cluster_service = None
            app.state.relation_graph_service = None
            app.state.import_service = BrowserImportService(
                settings=resolved_settings,
                repository=asset_repo,
                runner=pipeline_runner,
                durable_task_submitter=durable_task_submitter,
            )
            try:
                yield
            finally:
                if durable_task_supervisor is not None:
                    await durable_task_supervisor.close()
                await database.dispose()
            return

        embedding_client = DoubaoClient(resolved_settings)
        understanding_image_cache = ModelImageCache(
            target_bytes=resolved_settings.model_image_target_bytes,
            max_edge=resolved_settings.model_image_max_edge,
            fixed_size=resolved_settings.understanding_image_size,
            max_entries=resolved_settings.model_image_cache_entries,
        )
        embedding_image_cache = ModelImageCache(
            target_bytes=resolved_settings.model_image_target_bytes,
            max_edge=resolved_settings.model_image_max_edge,
            max_entries=resolved_settings.model_image_cache_entries,
        )
        video_frame_extractor = FFmpegVideoFrameExtractor(
            concurrency=resolved_settings.ffmpeg_concurrency
        )
        embedding_service = AssetEmbeddingService(
            settings=resolved_settings,
            repository=embedding_repository,
            model_client=embedding_client,
            vector_store=vectors,
            artifact_reader=storage,
            image_cache=embedding_image_cache,
            video_frame_extractor=video_frame_extractor,
        )
        understanding_service = AssetUnderstandingService(
            settings=resolved_settings,
            embedding_repository=embedding_repository,
            asset_repository=asset_repo,
            model_client=embedding_client,
            artifact_reader=storage,
            image_cache=understanding_image_cache,
            video_frame_extractor=video_frame_extractor,
        )
        search_repository = PostgresAssetSearchRepository(database)
        app.state.search_service = SearchService(
            query_embedding=QueryEmbeddingService(
                embedding_client,
                resolved_settings,
            ),
            recall=MultiChannelRecall(vectors, resolved_settings),
            assets=search_repository,
            text_recall=search_repository,
            clusters=search_repository,
            query_parser=QueryParser(embedding_client),
            history=history,
            image_resolver=query_images,
            search_vector_preparer=embedding_service,
            settings=resolved_settings,
        )
        cluster_service_instance = ClusterService(
            settings=resolved_settings,
            embedding_repository=embedding_repository,
            cluster_repository=cluster_repo,
            current_cluster_repository=current_cluster_repo,
            vector_store=vectors,
            model_client=embedding_client,
        )
        app.state.cluster_service = cluster_service_instance
        hierarchy_checkpointer = entity_hierarchy_checkpointer
        owned_checkpointer_context: Any | None = None
        if hierarchy_checkpointer is None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            owned_checkpointer_context = AsyncPostgresSaver.from_conn_string(
                _hierarchy_checkpoint_url(resolved_settings.database_url)
            )
            hierarchy_checkpointer = await owned_checkpointer_context.__aenter__()
            await hierarchy_checkpointer.setup()
        relation_graph_service_instance = RelationGraphService(
            embedding_repository=embedding_repository,
            understanding_service=understanding_service,
            model_client=embedding_client,
            current_cluster_repository=current_cluster_repo,
            relation_repository=relation_graph_repository,
            vector_store=vectors,
            incremental_entity_recall_similarity_threshold=(
                resolved_settings.relation_incremental_entity_recall_similarity_threshold
            ),
            incremental_entity_recall_top_k=(
                resolved_settings.relation_incremental_entity_recall_top_k
            ),
            hierarchy_checkpointer=hierarchy_checkpointer,
        )
        app.state.relation_graph_service = relation_graph_service_instance
        app.state.entity_hierarchy_checkpointer = hierarchy_checkpointer
        assignment_threshold = resolved_settings.cluster_incremental_assignment_threshold
        incremental_coordinator = IncrementalClusterCoordinator(
            settings=resolved_settings,
            assignment_service=IncrementalClusterService(
                repository=current_cluster_repo,
                vector_store=vectors,
                default_thresholds=IncrementalAssignmentThresholds(
                    resident_open=assignment_threshold,
                    dynamic=assignment_threshold,
                ),
            ),
            repository=current_cluster_repo,
            cluster_runner=cluster_service_instance,
            relation_graph_updater=relation_graph_service_instance,
        )
        app.state.incremental_cluster_coordinator = incremental_coordinator
        app.state.import_service = BrowserImportService(
            settings=resolved_settings,
            repository=asset_repo,
            runner=pipeline_runner,
            understanding_service=understanding_service,
            embedding_service=embedding_service,
            incremental_cluster_processor=incremental_coordinator,
            durable_task_submitter=durable_task_submitter,
        )
        import_workflow_coordinator = ImportWorkflowCoordinator(
            repository=asset_repo,
            understanding_service=understanding_service,
            embedding_service=embedding_service,
            incremental_cluster_processor=incremental_coordinator,
        )
        import_workflow_task = asyncio.create_task(import_workflow_coordinator.run_forever())
        try:
            yield
        finally:
            await import_workflow_coordinator.close()
            import_workflow_task.cancel()
            await asyncio.gather(import_workflow_task, return_exceptions=True)
            await incremental_coordinator.close()
            await embedding_client.close()
            if durable_task_supervisor is not None:
                await durable_task_supervisor.close()
            if owned_checkpointer_context is not None:
                await owned_checkpointer_context.__aexit__(None, None, None)
            await database.dispose()

    logging.basicConfig(
        level=getattr(logging, resolved_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    application = FastAPI(
        title="Capsule Search API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.search_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=[
            "Accept-Ranges",
            "Content-Range",
            "Content-Length",
            "ETag",
        ],
    )
    application.include_router(search_router)
    application.include_router(assets_router)
    application.include_router(capsules_router)
    application.include_router(cluster_runs_router)
    application.include_router(imports_router)
    application.include_router(workspaces_router)
    application.include_router(relation_graphs_router)

    @application.get("/health")
    async def health() -> dict[str, Any]:
        supervisor = getattr(application.state, "durable_task_supervisor", None)
        components = (
            {name: asdict(component) for name, component in supervisor.health_snapshot().items()}
            if supervisor is not None
            else {}
        )
        processing_ready = not components or all(
            component["ready"] and component["running"] for component in components.values()
        )
        return {
            "status": "ok" if processing_ready else "degraded",
            "search_ready": resolved_settings.ark_api_key is not None,
            "cluster_ready": resolved_settings.ark_api_key is not None,
            "processing_tasks": {
                "mode": getattr(
                    application.state,
                    "durable_task_runtime_mode",
                    "external",
                ),
                "ready": processing_ready,
                "components": components,
            },
        }

    return application


def _hierarchy_checkpoint_url(database_url: str) -> str:
    """Convert SQLAlchemy's asyncpg URL to the Psycopg URL used by LangGraph."""

    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


app = create_app()
