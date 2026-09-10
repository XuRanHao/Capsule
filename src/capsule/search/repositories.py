from collections.abc import Mapping, Sequence

from sqlalchemy import and_, func, select

from capsule.db.models import (
    Asset,
    EmbeddingRecord,
    SourceFile,
    UserFavorite,
)
from capsule.db.session import Database
from capsule.enums import EmbeddingStatus, ProcessingStatus
from capsule.search.models import (
    SearchAssetRecord,
    SearchFilters,
    TextSearchHit,
)


class PostgresAssetSearchRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
        embedding_ids: Sequence[str],
        created_by: str = "user_demo",
        filters: SearchFilters | None = None,
    ) -> Mapping[str, SearchAssetRecord]:
        if not asset_ids:
            return {}
        filters = filters or SearchFilters()
        statement = (
            select(Asset, SourceFile)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.workspace_id == workspace_id,
                SourceFile.workspace_id == workspace_id,
                Asset.asset_id.in_(asset_ids),
                Asset.generation == SourceFile.processing_generation,
                # Parent Assets are context containers, not independently
                # retrievable units. Exclude stale or mistakenly indexed
                # parent vectors during PostgreSQL hydration.
                Asset.index_role != "parent",
                # Timeline-only logical video segments deliberately opt out of
                # enrichment/vector retrieval; suppress any historical vectors.
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
            )
        )
        if filters.project_id:
            statement = statement.where(Asset.project_id == filters.project_id)
        if filters.asset_type:
            statement = statement.where(
                Asset.asset_type.in_([item.value for item in filters.asset_type])
            )
        if filters.file_type:
            statement = statement.where(Asset.file_type.in_(filters.file_type))
        if filters.source_file_id:
            statement = statement.where(Asset.source_file_id.in_(filters.source_file_id))
        if filters.created_at_from:
            statement = statement.where(Asset.created_at >= filters.created_at_from)
        if filters.created_at_to:
            statement = statement.where(Asset.created_at <= filters.created_at_to)
        if filters.favorite is not None:
            statement = statement.outerjoin(
                UserFavorite,
                and_(
                    UserFavorite.asset_id == Asset.asset_id,
                    UserFavorite.workspace_id == workspace_id,
                    UserFavorite.created_by == created_by,
                ),
            )
            if filters.favorite:
                statement = statement.where(UserFavorite.favorite_id.is_not(None))
            else:
                statement = statement.where(UserFavorite.favorite_id.is_(None))
        async with self._database.session() as session, session.begin():
            rows = (await session.execute(statement)).all()
            indexed_embedding_ids: dict[str, set[str]] = {}
            if embedding_ids:
                embedding_statement = select(
                    EmbeddingRecord.embedding_id,
                    EmbeddingRecord.asset_id,
                ).where(
                    EmbeddingRecord.workspace_id == workspace_id,
                    EmbeddingRecord.embedding_id.in_(embedding_ids),
                    EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
                )
                if filters.model_name:
                    embedding_statement = embedding_statement.where(
                        EmbeddingRecord.model_name.in_(filters.model_name)
                    )
                embedding_rows = (await session.execute(embedding_statement)).all()
                for embedding_id, asset_id in embedding_rows:
                    indexed_embedding_ids.setdefault(asset_id, set()).add(embedding_id)
        return {
            asset.asset_id: SearchAssetRecord(
                asset_id=asset.asset_id,
                workspace_id=asset.workspace_id,
                project_id=asset.project_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                file_name=asset.file_name,
                file_type=asset.file_type,
                asset_key=asset.asset_key,
                content_hash=asset.content_hash,
                asset_name=asset.asset_name,
                asset_name_source=asset.asset_name_source,
                asset_description=asset.asset_description,
                asset_features=dict(asset.asset_features),
                file_tree_context=list(asset.file_tree_context),
                source_contexts=list(asset.source_contexts),
                file_info=dict(asset.file_info),
                source_locator=dict(asset.source_locator),
                raw_content=asset.raw_content,
                derived_file_uri=asset.derived_file_uri,
                preview_uri=asset.preview_uri,
                processing_status=asset.processing_status,
                feature_revision=asset.feature_revision,
                embedding_revision=asset.embedding_revision,
                error_message=asset.error_message,
                created_at=asset.created_at,
                updated_at=asset.updated_at,
                source_workspace_id=source_file.workspace_id,
                source_project_id=source_file.project_id,
                source_file_name=source_file.original_file_name,
                source_file_type=source_file.file_type,
                source_mime_type=source_file.mime_type,
                source_relative_path=source_file.relative_path,
                source_file_tree_context=list(source_file.file_tree_context),
                source_storage_uri=source_file.storage_uri,
                source_sha256=source_file.sha256,
                source_file_size_bytes=source_file.file_size_bytes,
                source_processing_status=source_file.processing_status,
                source_error_message=source_file.error_message,
                source_created_at=source_file.created_at,
                source_updated_at=source_file.updated_at,
                indexed_embedding_ids=frozenset(indexed_embedding_ids.get(asset.asset_id, set())),
                parent_asset_id=asset.parent_asset_id,
                index_role=asset.index_role,
                child_order=asset.child_order,
            )
            for asset, source_file in rows
        }

    async def get_children_by_parent_ids(
        self,
        *,
        workspace_id: str,
        parent_asset_ids: Sequence[str],
    ) -> Mapping[str, SearchAssetRecord]:
        """Load the active child blocks belonging to the requested parents.

        Parent IDs come from already-authorized search hits. The workspace and
        active source generation checks remain mandatory so context expansion
        cannot cross a tenant boundary or revive stale chunks.
        """
        if not parent_asset_ids:
            return {}
        statement = (
            select(Asset.asset_id)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.workspace_id == workspace_id,
                SourceFile.workspace_id == workspace_id,
                Asset.parent_asset_id.in_(parent_asset_ids),
                Asset.index_role == "child",
                Asset.generation == SourceFile.processing_generation,
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
            )
        )
        async with self._database.session() as session:
            asset_ids = list((await session.scalars(statement)).all())
        return await self.get_by_ids(
            workspace_id=workspace_id,
            asset_ids=asset_ids,
            embedding_ids=(),
        )

    async def search_text(
        self,
        *,
        workspace_id: str,
        query_text: str,
        filters: SearchFilters,
        created_by: str,
        limit: int,
    ) -> Sequence[TextSearchHit]:
        """Recall indexed Asset text through ParadeDB's pg_search BM25 index."""

        normalized_query = query_text.strip()
        if limit < 1 or not normalized_query:
            return []
        statement = (
            select(
                Asset,
                func.pdb.score(Asset.asset_id).label("bm25_score"),
            )
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.asset_id.op("@@@")(normalized_query),
                Asset.workspace_id == workspace_id,
                SourceFile.workspace_id == workspace_id,
                Asset.generation == SourceFile.processing_generation,
                Asset.index_role != "parent",
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
            )
        )
        if filters.project_id:
            statement = statement.where(Asset.project_id == filters.project_id)
        if filters.asset_type:
            statement = statement.where(
                Asset.asset_type.in_([item.value for item in filters.asset_type])
            )
        if filters.file_type:
            statement = statement.where(Asset.file_type.in_(filters.file_type))
        if filters.source_file_id:
            statement = statement.where(Asset.source_file_id.in_(filters.source_file_id))
        if filters.created_at_from:
            statement = statement.where(Asset.created_at >= filters.created_at_from)
        if filters.created_at_to:
            statement = statement.where(Asset.created_at <= filters.created_at_to)
        if filters.favorite is not None:
            statement = statement.outerjoin(
                UserFavorite,
                and_(
                    UserFavorite.asset_id == Asset.asset_id,
                    UserFavorite.workspace_id == workspace_id,
                    UserFavorite.created_by == created_by,
                ),
            )
            statement = statement.where(
                UserFavorite.favorite_id.is_not(None)
                if filters.favorite
                else UserFavorite.favorite_id.is_(None)
            )
        if filters.model_name:
            statement = statement.where(
                Asset.asset_id.in_(
                    select(EmbeddingRecord.asset_id).where(
                        EmbeddingRecord.workspace_id == workspace_id,
                        EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
                        EmbeddingRecord.model_name.in_(filters.model_name),
                    )
                )
            )
        statement = statement.order_by(
            func.pdb.score(Asset.asset_id).desc(),
            Asset.asset_id.asc(),
        ).limit(limit)
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        return [
            TextSearchHit(
                asset_id=asset.asset_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                score=float(score),
            )
            for asset, score in rows
        ]
