from collections.abc import Mapping, Sequence

from sqlalchemy import select

from capsule.db.models import Asset, SourceFile
from capsule.db.session import Database
from capsule.search.models import SearchAssetRecord


class PostgresAssetSearchRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_by_ids(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
    ) -> Mapping[str, SearchAssetRecord]:
        if not asset_ids:
            return {}
        statement = (
            select(Asset, SourceFile)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.workspace_id == workspace_id,
                Asset.asset_id.in_(asset_ids),
            )
        )
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
            )
            for asset, source_file in rows
        }
