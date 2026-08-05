import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from capsule.api.assets import router as assets_router
from capsule.api.capsules import router as capsules_router
from capsule.api.clusters import router as cluster_runs_router
from capsule.api.imports import router as imports_router
from capsule.api.search import router as search_router
from capsule.config import Settings, get_settings
from capsule.db.repositories import (
    AssetRepository,
    ClusterRepository,
    CurrentClusterRepository,
    EmbeddingRepository,
)
from capsule.db.session import Database
from capsule.media.model_image import ModelImageCache
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.cluster_service import ClusterService
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.import_service import BrowserImportService
from capsule.pipeline.incremental_clustering import (
    IncrementalAssignmentThresholds,
    IncrementalClusterCoordinator,
    IncrementalClusterService,
)
from capsule.pipeline.runner import PipelineRunner
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.pipeline.workspace_clear import LibraryClearService
from capsule.search.history import SearchHistoryRepository
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.search.rerank import SearchReranker
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
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        if (
            search_service is not None
            or cluster_service is not None
            or cluster_repository is not None
            or current_cluster_repository is not None
            or import_service is not None
            or asset_repository is not None
            or library_clear_service is not None
        ):
            app.state.search_service = search_service
            app.state.cluster_service = cluster_service
            app.state.cluster_repository = cluster_repository
            if current_cluster_repository is not None:
                app.state.current_cluster_repository = current_cluster_repository
            app.state.import_service = import_service
            app.state.asset_repository = asset_repository
            app.state.library_clear_service = library_clear_service
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
        vectors = MilvusVectorStore(resolved_settings)
        pipeline_runner = PipelineRunner(
            settings=resolved_settings,
            database=database,
            object_storage=storage,
        )
        app.state.cluster_repository = cluster_repo
        app.state.current_cluster_repository = current_cluster_repo
        app.state.asset_repository = asset_repo
        app.state.library_clear_service = LibraryClearService(
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
            app.state.import_service = BrowserImportService(
                settings=resolved_settings,
                repository=asset_repo,
                runner=pipeline_runner,
            )
            try:
                yield
            finally:
                await database.dispose()
            return

        embedding_client = DoubaoClient(resolved_settings)
        model_image_cache = ModelImageCache(
            target_bytes=resolved_settings.model_image_target_bytes,
            max_edge=resolved_settings.model_image_max_edge,
            max_entries=resolved_settings.model_image_cache_entries,
        )
        embedding_service = AssetEmbeddingService(
            settings=resolved_settings,
            repository=embedding_repository,
            model_client=embedding_client,
            vector_store=vectors,
            artifact_reader=storage,
            image_cache=model_image_cache,
        )
        understanding_service = AssetUnderstandingService(
            settings=resolved_settings,
            embedding_repository=embedding_repository,
            asset_repository=asset_repo,
            model_client=embedding_client,
            artifact_reader=storage,
            image_cache=model_image_cache,
        )
        search_repository = PostgresAssetSearchRepository(database)
        app.state.search_service = SearchService(
            query_embedding=QueryEmbeddingService(
                embedding_client,
                resolved_settings,
            ),
            recall=MultiChannelRecall(vectors, resolved_settings),
            assets=search_repository,
            clusters=search_repository,
            query_parser=QueryParser(embedding_client),
            reranker=SearchReranker(embedding_client),
            history=history,
            image_resolver=query_images,
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
        )
        app.state.incremental_cluster_coordinator = incremental_coordinator
        app.state.import_service = BrowserImportService(
            settings=resolved_settings,
            repository=asset_repo,
            runner=pipeline_runner,
            understanding_service=understanding_service,
            embedding_service=embedding_service,
            incremental_cluster_processor=incremental_coordinator,
        )
        try:
            yield
        finally:
            await incremental_coordinator.close()
            await embedding_client.close()
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
        allow_headers=["Content-Type"],
    )
    application.include_router(search_router)
    application.include_router(assets_router)
    application.include_router(capsules_router)
    application.include_router(cluster_runs_router)
    application.include_router(imports_router)

    @application.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "search_ready": resolved_settings.ark_api_key is not None,
            "cluster_ready": resolved_settings.ark_api_key is not None,
        }

    return application


app = create_app()
