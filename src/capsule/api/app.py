import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from capsule.api.capsules import router as capsules_router
from capsule.api.search import router as search_router
from capsule.config import Settings, get_settings
from capsule.db.session import Database
from capsule.model_clients.doubao import DoubaoClient
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
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved_settings
        if search_service is not None:
            app.state.search_service = search_service
            yield
            return

        database = Database(resolved_settings)
        storage = ObjectStorage(resolved_settings)
        await storage.ensure_bucket()
        history = SearchHistoryRepository(database, resolved_settings)
        query_images = QueryImageService(database, storage)
        app.state.search_history = history
        app.state.query_image_service = query_images
        if resolved_settings.ark_api_key is None:
            logging.getLogger(__name__).warning(
                "CAPSULE_ARK_API_KEY is not configured; search endpoint will return 503"
            )
            app.state.search_service = None
            try:
                yield
            finally:
                await database.dispose()
            return

        embedding_client = DoubaoClient(resolved_settings)
        vectors = MilvusVectorStore(resolved_settings)
        app.state.search_service = SearchService(
            query_embedding=QueryEmbeddingService(
                embedding_client,
                resolved_settings,
            ),
            recall=MultiChannelRecall(vectors, resolved_settings),
            assets=PostgresAssetSearchRepository(database),
            query_parser=QueryParser(embedding_client),
            reranker=SearchReranker(embedding_client),
            history=history,
            image_resolver=query_images,
            settings=resolved_settings,
        )
        try:
            yield
        finally:
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
    application.include_router(capsules_router)

    @application.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "search_ready": resolved_settings.ark_api_key is not None,
        }

    return application


app = create_app()
