from collections.abc import Mapping, Sequence

from sqlalchemy import and_, select

from capsule.db.models import Asset, ClusterMembership, SourceFile, UserFavorite
from capsule.db.session import Database
from capsule.search.models import SearchAssetRecord, SearchFilters


class PostgresAssetSearchRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
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
                Asset.asset_id.in_(asset_ids),
            )
        )
        if filters.project_id:
            statement = statement.where(Asset.project_id == filters.project_id)
        if filters.asset_type:
            statement = statement.where(
                Asset.asset_type.in_([item.value for item in filters.asset_type])
            )
        if filters.file_type:
            statement = statement.where(SourceFile.file_type.in_(filters.file_type))
        if filters.source_file_id:
            statement = statement.where(Asset.source_file_id.in_(filters.source_file_id))
        if filters.created_at_from:
            statement = statement.where(Asset.created_at >= filters.created_at_from)
        if filters.created_at_to:
            statement = statement.where(Asset.created_at <= filters.created_at_to)
        if filters.cluster_capsule_id:
            statement = statement.join(
                ClusterMembership,
                ClusterMembership.asset_id == Asset.asset_id,
            ).where(ClusterMembership.cluster_capsule_id == filters.cluster_capsule_id)
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
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return {
            asset.asset_id: SearchAssetRecord(
                asset_id=asset.asset_id,
                workspace_id=asset.workspace_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                asset_name=asset.asset_name,
                asset_description=asset.asset_description,
                asset_features=asset.asset_features,
                source_contexts=asset.source_contexts,
                source_locator=asset.source_locator,
                preview_uri=asset.preview_uri,
                processing_status=asset.processing_status,
                source_file_name=source_file.original_file_name,
                source_file_type=source_file.file_type,
                source_relative_path=source_file.relative_path,
                project_id=asset.project_id,
                created_at=asset.created_at,
            )
            for asset, source_file in rows
        }
