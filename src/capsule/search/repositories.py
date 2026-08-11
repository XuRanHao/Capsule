import re
from collections.abc import Mapping, Sequence

from sqlalchemy import and_, func, or_, select

from capsule.db.models import (
    Asset,
    CurrentCluster,
    CurrentClusterMember,
    EmbeddingRecord,
    SourceFile,
    UserFavorite,
)
from capsule.db.session import Database
from capsule.enums import EmbeddingStatus, EmbeddingType, ProcessingStatus
from capsule.search.models import (
    ClusterSearchResult,
    SearchAssetRecord,
    SearchFilters,
    TextSearchHit,
)

_TEXT_PART_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


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
        if filters.cluster_capsule_id:
            statement = statement.join(
                CurrentClusterMember,
                CurrentClusterMember.asset_id == Asset.asset_id,
            ).where(CurrentClusterMember.cluster_id == filters.cluster_capsule_id)
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
        """Recall filename, path, raw text and description as one local route."""

        if limit < 1:
            return []
        terms = _text_search_terms(query_text)
        if not terms:
            return []
        text_fields = (
            Asset.file_name,
            SourceFile.relative_path,
            Asset.raw_content,
            Asset.asset_description,
        )
        statement = (
            select(Asset, SourceFile)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.workspace_id == workspace_id,
                SourceFile.workspace_id == workspace_id,
                Asset.generation == SourceFile.processing_generation,
                Asset.index_role != "parent",
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
                or_(
                    *(
                        field.ilike(_contains_pattern(term), escape="\\")
                        for term in terms
                        for field in text_fields
                    )
                ),
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
        if filters.cluster_capsule_id:
            statement = statement.join(
                CurrentClusterMember,
                CurrentClusterMember.asset_id == Asset.asset_id,
            ).where(CurrentClusterMember.cluster_id == filters.cluster_capsule_id)
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
        candidate_limit = min(max(limit * 20, 200), 2_000)
        statement = statement.order_by(Asset.updated_at.desc()).limit(candidate_limit)
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        hits = [
            TextSearchHit(
                asset_id=asset.asset_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                score=_text_relevance(
                    query_text=query_text,
                    terms=terms,
                    file_name=asset.file_name,
                    relative_path=source.relative_path,
                    raw_content=asset.raw_content,
                    asset_description=asset.asset_description,
                ),
            )
            for asset, source in rows
        ]
        hits = [item for item in hits if item.score > 0]
        hits.sort(key=lambda item: (-item.score, item.asset_id))
        return hits[:limit]

    async def search_by_assets(
        self,
        *,
        workspace_id: str,
        asset_scores: Mapping[str, float],
        embedding_types: Sequence[str],
        limit: int,
    ) -> Sequence[ClusterSearchResult]:
        """Aggregate matched Assets through the currently published cluster state."""
        if not asset_scores or not embedding_types or limit < 1:
            return []

        member_stats = (
            select(
                CurrentClusterMember.cluster_id,
                func.count(CurrentClusterMember.asset_id).label("member_count"),
                func.avg(func.coalesce(CurrentClusterMember.score, 1.0)).label("average_score"),
            )
            .group_by(CurrentClusterMember.cluster_id)
            .subquery()
        )
        statement = (
            select(
                CurrentClusterMember.asset_id,
                func.coalesce(CurrentClusterMember.score, 1.0),
                CurrentCluster,
                member_stats.c.member_count,
                member_stats.c.average_score,
            )
            .join(
                CurrentCluster,
                CurrentCluster.cluster_id == CurrentClusterMember.cluster_id,
            )
            .join(
                member_stats,
                member_stats.c.cluster_id == CurrentCluster.cluster_id,
            )
            .where(
                CurrentClusterMember.asset_id.in_(asset_scores),
                CurrentCluster.workspace_id == workspace_id,
                CurrentCluster.embedding_type.in_(set(embedding_types)),
            )
        )
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        maximum_asset_score = max(asset_scores.values(), default=0.0)
        if maximum_asset_score <= 0:
            return []
        grouped: dict[
            str,
            tuple[CurrentCluster, int, float, list[tuple[str, float]]],
        ] = {}
        for asset_id, membership_probability, cluster, member_count, average_score in rows:
            normalized_asset_score = asset_scores.get(asset_id, 0.0) / maximum_asset_score
            membership_weighted_score = normalized_asset_score * (
                0.75 + 0.25 * float(membership_probability)
            )
            group = grouped.setdefault(
                cluster.cluster_id,
                (cluster, int(member_count), float(average_score), []),
            )
            group[3].append((asset_id, membership_weighted_score))

        results: list[ClusterSearchResult] = []
        for cluster, member_count, average_score, matches in grouped.values():
            matches.sort(key=lambda item: (-item[1], item[0]))
            match_scores = [item[1] for item in matches]
            coverage = min(1.0, len(matches) / max(1, member_count))
            score = (
                0.65 * max(match_scores)
                + 0.25 * (sum(match_scores) / len(match_scores))
                + 0.10 * coverage
            )
            results.append(
                ClusterSearchResult(
                    cluster_capsule_id=cluster.cluster_id,
                    cluster_run_id=cluster.source_run_id or "",
                    embedding_type=EmbeddingType(cluster.embedding_type),
                    name=cluster.name,
                    description=cluster.description,
                    keywords=[],
                    common_features=[],
                    member_count=member_count,
                    average_membership_probability=average_score,
                    medoid_asset_id=cluster.representative_asset_id,
                    representative_asset_ids=(
                        [cluster.representative_asset_id]
                        if cluster.representative_asset_id is not None
                        else []
                    ),
                    matched_asset_ids=[item[0] for item in matches],
                    matched_asset_count=len(matches),
                    score=score,
                )
            )
        results.sort(key=lambda item: (-item.score, item.cluster_capsule_id))
        return results[:limit]


def _text_search_terms(query_text: str, *, limit: int = 24) -> tuple[str, ...]:
    normalized = " ".join(query_text.casefold().split())
    candidates: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> bool:
        if term not in seen:
            seen.add(term)
            candidates.append(term)
        return len(candidates) >= limit

    if 1 < len(normalized) <= 128 and add(normalized):
        return tuple(candidates)
    for part in _TEXT_PART_RE.findall(normalized[:2_000]):
        if len(part) < 2:
            continue
        if _CJK_RE.search(part):
            if len(part) <= 8 and add(part):
                break
            if any(add(part[index : index + 2]) for index in range(len(part) - 1)):
                break
        elif add(part):
            break
    return tuple(candidates)


def _contains_pattern(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _text_relevance(
    *,
    query_text: str,
    terms: Sequence[str],
    file_name: str | None,
    relative_path: str | None,
    raw_content: str | None,
    asset_description: str | None,
) -> float:
    normalized_query = " ".join(query_text.casefold().split())
    best = 0.0
    fields = (
        (file_name, 1.0),
        (relative_path, 0.9),
        (raw_content, 0.8),
        (asset_description, 0.75),
    )
    for value, weight in fields:
        normalized = value.casefold() if value else ""
        if not normalized:
            continue
        matched = sum(term in normalized for term in terms)
        if not matched:
            continue
        coverage = matched / len(terms)
        full_match = bool(normalized_query and normalized_query in normalized)
        exact_match = bool(normalized_query and normalized.strip() == normalized_query)
        field_score = weight * (
            0.2 + 0.45 * coverage + 0.2 * full_match + 0.15 * exact_match
        )
        best = max(best, field_score)
    return min(1.0, best)
