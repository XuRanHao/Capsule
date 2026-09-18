import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, cast
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from capsule.agent.checkpoints import postgres_checkpoint_url
from capsule.agent.context_budget import ContextBudget
from capsule.agent.graph_tools import build_graph_tool_registry
from capsule.agent.memory_worker import MemoryOutboxDispatcher, RedisMemoryQueue
from capsule.agent.milvus_memory_store import MilvusAgentMemoryStore
from capsule.agent.model_planner import ModelAgentPlanner
from capsule.agent.runtime import AgentRuntime
from capsule.agent.tools import ToolExecutionStore
from capsule.api.agent import router as agent_router
from capsule.api.assets import router as assets_router
from capsule.api.capsules import router as capsules_router
from capsule.api.graphs import router as graphs_router
from capsule.api.imports import router as imports_router
from capsule.api.search import router as search_router
from capsule.api.workspaces import router as workspaces_router
from capsule.config import Settings, get_settings
from capsule.db.agent_memory import (
    AgentConversationRepository,
    MemoryOutboxEvent,
    PostgresAgentMemoryStore,
)
from capsule.db.repositories import (
    AgentToolExecutionRepository,
    AssetRepository,
    EmbeddingRepository,
    RelationGraphRepository,
    WorkspaceUserRepository,
)
from capsule.db.session import Database
from capsule.db.workspace_directories import WorkspaceDirectoryRepository
from capsule.media.model_image import ModelImageCache
from capsule.media.video_frames import FFmpegVideoFrameExtractor
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.durable_task_supervisor import DurableTaskRuntimeSupervisor
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.import_service import BrowserImportService, ImportWorkflowCoordinator
from capsule.pipeline.processing_task_service import BrowserProcessingTaskSubmissionService
from capsule.pipeline.runner import PipelineRunner
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.pipeline.workspace_clear import LibraryClearService
from capsule.pipeline.workspace_directories import WorkspaceDirectoryService
from capsule.pipeline.workspace_management import WorkspaceService
from capsule.search.history import SearchHistoryRepository
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.search.service import SearchService
from capsule.search.uploads import QueryImageService
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.agent_memory import AgentMemoryMilvusStore
from capsule.vectorstore.milvus import MilvusVectorStore


def create_app(
    *,
    settings: Settings | None = None,
    search_service: SearchService | None = None,
    import_service: BrowserImportService | None = None,
    asset_repository: AssetRepository | None = None,
    library_clear_service: LibraryClearService | None = None,
    workspace_service: WorkspaceService | None = None,
    workspace_directory_service: WorkspaceDirectoryService | None = None,
    agent_runtime: AgentRuntime | None = None,
    graph_repository: RelationGraphRepository | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_agent_runtime = agent_runtime or AgentRuntime()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        app.state.agent_runtime = resolved_agent_runtime
        app.state.relation_graph_repository = graph_repository
        app.state.video_transcode_semaphore = asyncio.Semaphore(
            resolved_settings.video_transcode_concurrency
        )
        if (
            search_service is not None
            or import_service is not None
            or asset_repository is not None
            or library_clear_service is not None
            or workspace_service is not None
            or workspace_directory_service is not None
            or agent_runtime is not None
            or graph_repository is not None
        ):
            app.state.search_service = search_service
            app.state.import_service = import_service
            app.state.asset_repository = asset_repository
            app.state.library_clear_service = library_clear_service
            app.state.workspace_service = workspace_service
            app.state.workspace_directory_service = workspace_directory_service
            yield
            return

        database = Database(resolved_settings)
        app.state.agent_conversation_repository = AgentConversationRepository(database)
        relation_graph_repository = RelationGraphRepository(database)
        app.state.relation_graph_repository = relation_graph_repository
        memory_context_queue: RedisMemoryQueue | None = None
        memory_event_publisher = None
        try:
            memory_context_queue = RedisMemoryQueue(
                redis_url=resolved_settings.redis_url,
                stream=resolved_settings.agent_memory_stream,
                group=resolved_settings.agent_memory_group,
                consumer=f"api-context-{uuid4().hex[:12]}",
            )
            await memory_context_queue.start()
            memory_dispatcher = MemoryOutboxDispatcher(
                repository=app.state.agent_conversation_repository,
                queue=memory_context_queue,
            )

            async def publish_memory_event(event: MemoryOutboxEvent) -> None:
                # The outbox remains recoverable by the external dispatcher;
                # this just removes one polling interval from a blocked turn.
                await memory_dispatcher.dispatch_event(event)

            memory_event_publisher = publish_memory_event
        except Exception:
            if memory_context_queue is not None:
                await memory_context_queue.close()
            memory_context_queue = None
            logging.getLogger(__name__).warning(
                "Agent memory context publisher is unavailable; "
                "external outbox dispatch will be used",
                exc_info=True,
            )
        app.state.agent_memory_context_publisher_ready = memory_event_publisher is not None
        if agent_runtime is None:
            _configure_default_agent_runtime(
                runtime=resolved_agent_runtime,
                database=database,
                conversation_repository=app.state.agent_conversation_repository,
                graph_repository=relation_graph_repository,
                settings=resolved_settings,
                memory_event_publisher=memory_event_publisher,
            )
        storage = ObjectStorage(resolved_settings)
        await storage.ensure_bucket()
        history = SearchHistoryRepository(database, resolved_settings)
        query_images = QueryImageService(database, storage)
        app.state.search_history = history
        app.state.query_image_service = query_images
        app.state.object_storage = storage
        asset_repo = AssetRepository(database)
        embedding_repository = EmbeddingRepository(database)
        vectors = MilvusVectorStore(resolved_settings)
        memory_vectors = AgentMemoryMilvusStore(resolved_settings)
        app.state.agent_memory_vector_store = memory_vectors
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
            memory_vector_store=memory_vectors,
            object_storage=storage,
        )
        app.state.workspace_directory_service = WorkspaceDirectoryService(
            repository=WorkspaceDirectoryRepository(database)
        )
        if resolved_settings.ark_api_key is None:
            if agent_runtime is None:
                resolved_agent_runtime.set_memory_store(
                    PostgresAgentMemoryStore(
                        app.state.agent_conversation_repository,
                        per_scope_limit=resolved_settings.agent_memory_context_per_scope,
                    )
                )
            logging.getLogger(__name__).warning(
                "CAPSULE_ARK_API_KEY is not configured; search endpoint will return 503"
            )
            app.state.search_service = None
            app.state.import_service = BrowserImportService(
                settings=resolved_settings,
                repository=asset_repo,
                runner=pipeline_runner,
                durable_task_submitter=durable_task_submitter,
            )
            checkpoint_context = None
            if agent_runtime is None:
                checkpointer, checkpoint_context = await _open_agent_checkpointer(
                    settings=resolved_settings,
                    runtime=resolved_agent_runtime,
                )
                app.state.agent_checkpointer = checkpointer
            try:
                yield
            finally:
                if durable_task_supervisor is not None:
                    await durable_task_supervisor.close()
                if memory_context_queue is not None:
                    await memory_context_queue.close()
                await database.dispose()
                if checkpoint_context is not None:
                    await checkpoint_context.__aexit__(None, None, None)
            return

        embedding_client = DoubaoClient(resolved_settings)
        if agent_runtime is None:
            resolved_agent_runtime.set_planner(
                ModelAgentPlanner(
                    model=embedding_client,
                    model_name=resolved_settings.agent_planner_model,
                    max_output_tokens=resolved_settings.agent_planner_max_output_tokens,
                )
            )
            resolved_agent_runtime.set_memory_store(
                MilvusAgentMemoryStore(
                    repository=app.state.agent_conversation_repository,
                    embedder=embedding_client,
                    vector_store=memory_vectors,
                    per_scope_limit=resolved_settings.agent_memory_context_per_scope,
                    candidate_multiplier=(
                        resolved_settings.agent_memory_vector_candidate_multiplier
                    ),
                )
            )
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
            query_parser=QueryParser(embedding_client),
            history=history,
            image_resolver=query_images,
            search_vector_preparer=embedding_service,
            settings=resolved_settings,
        )
        app.state.import_service = BrowserImportService(
            settings=resolved_settings,
            repository=asset_repo,
            runner=pipeline_runner,
            understanding_service=understanding_service,
            embedding_service=embedding_service,
            durable_task_submitter=durable_task_submitter,
        )
        import_workflow_coordinator = ImportWorkflowCoordinator(
            repository=asset_repo,
            understanding_service=understanding_service,
            embedding_service=embedding_service,
        )
        import_workflow_task = asyncio.create_task(import_workflow_coordinator.run_forever())
        checkpoint_context = None
        if agent_runtime is None:
            checkpointer, checkpoint_context = await _open_agent_checkpointer(
                settings=resolved_settings,
                runtime=resolved_agent_runtime,
            )
            app.state.agent_checkpointer = checkpointer
        try:
            yield
        finally:
            await import_workflow_coordinator.close()
            import_workflow_task.cancel()
            await asyncio.gather(import_workflow_task, return_exceptions=True)
            await embedding_client.close()
            if durable_task_supervisor is not None:
                await durable_task_supervisor.close()
            if memory_context_queue is not None:
                await memory_context_queue.close()
            await database.dispose()
            if checkpoint_context is not None:
                await checkpoint_context.__aexit__(None, None, None)

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
    application.include_router(agent_router)
    application.include_router(graphs_router)
    application.include_router(assets_router)
    application.include_router(capsules_router)
    application.include_router(imports_router)
    application.include_router(workspaces_router)

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


def _agent_context_budget(settings: Settings) -> ContextBudget:
    return ContextBudget(
        window_tokens=settings.agent_context_window_tokens,
        output_reserve_tokens=settings.agent_context_output_reserve_tokens,
        short_term_ratio=settings.agent_context_short_term_ratio,
        tool_result_ratio=settings.agent_context_tool_result_ratio,
        raw_tail_tokens=settings.agent_context_raw_tail_tokens,
        max_reduction_rounds=settings.agent_context_max_reduction_rounds,
        summary_wait_seconds=settings.agent_context_summary_wait_seconds,
        summary_poll_seconds=settings.agent_context_summary_poll_seconds,
    )


def _configure_default_agent_runtime(
    *,
    runtime: AgentRuntime,
    database: Database,
    conversation_repository: AgentConversationRepository,
    graph_repository: RelationGraphRepository,
    settings: Settings,
    memory_event_publisher: Any | None,
) -> None:
    """Attach all server-owned dependencies to the normal API Runtime."""

    runtime.set_permission_loader(WorkspaceUserRepository(database).load_granted_permissions)
    runtime.set_conversation_repository(
        conversation_repository,
        context_budget=_agent_context_budget(settings),
        memory_event_publisher=memory_event_publisher,
        turn_lease_seconds=settings.agent_turn_lease_seconds,
    )
    runtime.set_tools(
        build_graph_tool_registry(
            graph_repository,
            execution_store=cast(
                ToolExecutionStore,
                AgentToolExecutionRepository(database),
            ),
        )
    )

async def _open_agent_checkpointer(
    *,
    settings: Settings,
    runtime: AgentRuntime,
) -> tuple[AsyncPostgresSaver, Any]:
    """Start LangGraph's checkpoint schema beside Capsule's business tables."""

    context = AsyncPostgresSaver.from_conn_string(postgres_checkpoint_url(settings))
    checkpointer = await context.__aenter__()
    try:
        await checkpointer.setup()
    except BaseException as exc:
        await context.__aexit__(type(exc), exc, exc.__traceback__)
        raise
    runtime.set_checkpointer(checkpointer)
    return checkpointer, context


app = create_app()
