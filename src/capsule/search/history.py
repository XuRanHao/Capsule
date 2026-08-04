from datetime import UTC, datetime

from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.config import Settings
from capsule.db.models import (
    Asset,
    SearchCapsule,
    SearchExecution,
    SearchResultSnapshot,
)
from capsule.db.session import Database
from capsule.search.models import (
    CapsuleSnapshot,
    FusionMethod,
    ParsedQuery,
    QueryType,
    RerankMethod,
    SearchCapsuleDetail,
    SearchCapsuleListResponse,
    SearchCapsuleSummary,
    SearchFilters,
    SearchRequest,
    SearchResult,
)


class SearchCapsuleNotFoundError(LookupError):
    pass


class SearchHistoryRepository:
    def __init__(self, database: Database, settings: Settings) -> None:
        self._database = database
        self._settings = settings

    async def record_success(
        self,
        *,
        request: SearchRequest,
        parsed_query: ParsedQuery,
        results: list[SearchResult],
        degraded: bool,
        degraded_reasons: list[str],
        latency_ms: int,
        existing_capsule_id: str | None = None,
    ) -> tuple[str, str]:
        now = datetime.now(UTC)
        async with self._database.session() as session:
            capsule: SearchCapsule
            if existing_capsule_id is None:
                capsule = SearchCapsule(
                    workspace_id=request.workspace_id,
                    created_by=request.created_by,
                    query_type=request.query_type.value,
                    query_text=request.query_text,
                    query_image_uri=request.query_image_upload_id or request.query_image_url,
                    parsed_query=parsed_query.model_dump(mode="json"),
                    filters=request.filters.model_dump(mode="json"),
                    fusion_method=request.fusion_method.value,
                    rerank_method=request.rerank_method.value,
                    search_engine_version=self._settings.search_engine_version,
                    embedding_model=self._settings.embedding_model,
                    is_favorite=request.save_capsule,
                    last_used_at=now,
                )
                session.add(capsule)
                await session.flush()
            else:
                existing = await session.scalar(
                    select(SearchCapsule).where(
                        SearchCapsule.capsule_id == existing_capsule_id,
                        SearchCapsule.workspace_id == request.workspace_id,
                        SearchCapsule.created_by == request.created_by,
                    )
                )
                if existing is None:
                    raise SearchCapsuleNotFoundError(existing_capsule_id)
                capsule = existing
                capsule.last_used_at = now

            execution = SearchExecution(
                capsule_id=capsule.capsule_id,
                workspace_id=request.workspace_id,
                request_payload=request.model_dump(mode="json"),
                parsed_query=parsed_query.model_dump(mode="json"),
                status="completed",
                degraded=degraded,
                degraded_reasons=degraded_reasons,
                latency_ms=latency_ms,
            )
            session.add(execution)
            await session.flush()
            for rank, result in enumerate(results, start=1):
                session.add(
                    SearchResultSnapshot(
                        execution_id=execution.execution_id,
                        capsule_id=capsule.capsule_id,
                        asset_id=result.asset_id,
                        result_rank=rank,
                        final_score=result.score,
                        component_scores={
                            "rerank_score": result.rerank_score,
                            "channels": [
                                item.model_dump(mode="json") for item in result.matched_channels
                            ],
                        },
                        result_payload=result.model_dump(mode="json"),
                    )
                )
            await session.commit()
            capsule_id = capsule.capsule_id
            execution_id = execution.execution_id

        await self._prune_recent(
            workspace_id=request.workspace_id,
            created_by=request.created_by,
        )
        return capsule_id, execution_id

    async def list_capsules(
        self,
        *,
        workspace_id: str,
        created_by: str,
        favorites_only: bool = False,
        limit: int = 100,
    ) -> SearchCapsuleListResponse:
        statement = (
            select(SearchCapsule)
            .where(
                SearchCapsule.workspace_id == workspace_id,
                SearchCapsule.created_by == created_by,
            )
            .order_by(desc(SearchCapsule.is_favorite), desc(SearchCapsule.last_used_at))
            .limit(limit)
        )
        if favorites_only:
            statement = statement.where(SearchCapsule.is_favorite.is_(True))
        async with self._database.session() as session:
            capsules = list((await session.scalars(statement)).all())
            items = [await self._summary(session, capsule) for capsule in capsules]
        return SearchCapsuleListResponse(items=items)

    async def get_capsule(
        self,
        *,
        capsule_id: str,
        workspace_id: str,
        created_by: str,
    ) -> SearchCapsuleDetail:
        async with self._database.session() as session:
            capsule = await self._get_owned_capsule(
                session,
                capsule_id=capsule_id,
                workspace_id=workspace_id,
                created_by=created_by,
            )
            executions = list(
                (
                    await session.scalars(
                        select(SearchExecution)
                        .where(SearchExecution.capsule_id == capsule_id)
                        .order_by(desc(SearchExecution.created_at))
                    )
                ).all()
            )
            if not executions:
                raise SearchCapsuleNotFoundError(capsule_id)
            latest = executions[0]
            snapshot_rows = (
                await session.execute(
                    select(SearchResultSnapshot, Asset.asset_id)
                    .outerjoin(
                        Asset,
                        (Asset.asset_id == SearchResultSnapshot.asset_id)
                        & (Asset.workspace_id == workspace_id),
                    )
                    .where(SearchResultSnapshot.execution_id == latest.execution_id)
                    .order_by(SearchResultSnapshot.result_rank)
                )
            ).all()
            snapshot_results = [
                _restore_result(row, available_asset_id)
                for row, available_asset_id in snapshot_rows
            ]
            summary = await self._summary(session, capsule, result_count=len(snapshot_results))
            return SearchCapsuleDetail(
                **summary.model_dump(),
                parsed_query=ParsedQuery.model_validate(capsule.parsed_query),
                filters=SearchFilters.model_validate(capsule.filters),
                search_engine_version=capsule.search_engine_version,
                embedding_model=capsule.embedding_model,
                latest_snapshot=CapsuleSnapshot(
                    execution_id=latest.execution_id,
                    created_at=latest.created_at,
                    results=snapshot_results,
                ),
                executions=[item.execution_id for item in executions],
            )

    async def load_request(
        self,
        *,
        capsule_id: str,
        workspace_id: str,
        created_by: str,
    ) -> SearchRequest:
        async with self._database.session() as session:
            capsule = await self._get_owned_capsule(
                session,
                capsule_id=capsule_id,
                workspace_id=workspace_id,
                created_by=created_by,
            )
            latest = await session.scalar(
                select(SearchExecution)
                .where(SearchExecution.capsule_id == capsule.capsule_id)
                .order_by(desc(SearchExecution.created_at))
                .limit(1)
            )
            if latest is None:
                raise SearchCapsuleNotFoundError(capsule_id)
            request = SearchRequest.model_validate(latest.request_payload)
            return request.model_copy(
                update={
                    "save_capsule": False,
                    "workspace_id": workspace_id,
                    "created_by": created_by,
                }
            )

    async def set_favorite(
        self,
        *,
        capsule_id: str,
        workspace_id: str,
        created_by: str,
        is_favorite: bool,
    ) -> SearchCapsuleDetail:
        async with self._database.session() as session:
            capsule = await self._get_owned_capsule(
                session,
                capsule_id=capsule_id,
                workspace_id=workspace_id,
                created_by=created_by,
            )
            capsule.is_favorite = is_favorite
            capsule.last_used_at = datetime.now(UTC)
            await session.commit()
        return await self.get_capsule(
            capsule_id=capsule_id,
            workspace_id=workspace_id,
            created_by=created_by,
        )

    async def delete_capsule(
        self,
        *,
        capsule_id: str,
        workspace_id: str,
        created_by: str,
    ) -> None:
        async with self._database.session() as session:
            capsule = await self._get_owned_capsule(
                session,
                capsule_id=capsule_id,
                workspace_id=workspace_id,
                created_by=created_by,
            )
            await session.delete(capsule)
            await session.commit()

    async def _summary(
        self,
        session: AsyncSession,
        capsule: SearchCapsule,
        *,
        result_count: int | None = None,
    ) -> SearchCapsuleSummary:
        if result_count is None:
            latest_execution_id = await session.scalar(
                select(SearchExecution.execution_id)
                .where(SearchExecution.capsule_id == capsule.capsule_id)
                .order_by(desc(SearchExecution.created_at))
                .limit(1)
            )
            result_count = 0
            if latest_execution_id is not None:
                result_count = int(
                    await session.scalar(
                        select(func.count())
                        .select_from(SearchResultSnapshot)
                        .where(SearchResultSnapshot.execution_id == latest_execution_id)
                    )
                    or 0
                )
        return SearchCapsuleSummary(
            capsule_id=capsule.capsule_id,
            workspace_id=capsule.workspace_id,
            created_by=capsule.created_by,
            query_type=QueryType(capsule.query_type),
            query_text=capsule.query_text,
            query_image_uri=capsule.query_image_uri,
            fusion_method=FusionMethod(capsule.fusion_method),
            rerank_method=RerankMethod(capsule.rerank_method),
            is_favorite=capsule.is_favorite,
            result_count=result_count,
            last_used_at=capsule.last_used_at,
            created_at=capsule.created_at,
        )

    async def _get_owned_capsule(
        self,
        session: AsyncSession,
        *,
        capsule_id: str,
        workspace_id: str,
        created_by: str,
    ) -> SearchCapsule:
        capsule = await session.scalar(
            select(SearchCapsule).where(
                SearchCapsule.capsule_id == capsule_id,
                SearchCapsule.workspace_id == workspace_id,
                SearchCapsule.created_by == created_by,
            )
        )
        if capsule is None:
            raise SearchCapsuleNotFoundError(capsule_id)
        return capsule

    async def _prune_recent(self, *, workspace_id: str, created_by: str) -> None:
        async with self._database.session() as session:
            expired_ids = list(
                (
                    await session.scalars(
                        select(SearchCapsule.capsule_id)
                        .where(
                            SearchCapsule.workspace_id == workspace_id,
                            SearchCapsule.created_by == created_by,
                            SearchCapsule.is_favorite.is_(False),
                        )
                        .order_by(desc(SearchCapsule.last_used_at))
                        .offset(self._settings.search_capsule_recent_limit)
                    )
                ).all()
            )
            if expired_ids:
                await session.execute(
                    delete(SearchCapsule).where(SearchCapsule.capsule_id.in_(expired_ids))
                )
                await session.commit()


def _restore_result(
    snapshot: SearchResultSnapshot,
    available_asset_id: str | None,
) -> SearchResult:
    result = SearchResult.model_validate(snapshot.result_payload)
    if available_asset_id is not None:
        return result
    return result.model_copy(
        update={
            "asset_name": "素材已删除或无权限",
            "asset_description": None,
            "asset_features": {},
            "source_contexts": [],
            "source_locator": {},
            "preview_uri": None,
            "source_file": None,
            "matched_feature": None,
            "matched_reason": "该快照项当前不可用",
            "available": False,
        }
    )
