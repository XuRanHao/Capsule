import os

import pytest

from capsule.config import Settings
from capsule.db.session import Database
from capsule.search.history import SearchHistoryRepository
from capsule.search.models import SearchRequest
from capsule.search.query_parser import QueryParser


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("CAPSULE_RUN_POSTGRES_INTEGRATION") != "1",
    reason="set CAPSULE_RUN_POSTGRES_INTEGRATION=1 to exercise Search Capsule persistence",
)
async def test_search_capsule_snapshot_refresh_state_and_favorite() -> None:
    settings = Settings()
    database = Database(settings)
    history = SearchHistoryRepository(database, settings)
    request = SearchRequest(
        workspace_id="workspace_demo",
        created_by="integration_test",
        query_type="text",
        query_text="蓝紫色黄昏",
    )
    parsed, _ = await QueryParser().parse(request, image_url=None)
    capsule_id = ""
    try:
        capsule_id, execution_id = await history.record_success(
            request=request,
            parsed_query=parsed,
            results=[],
            degraded=False,
            degraded_reasons=[],
            latency_ms=12,
        )
        detail = await history.get_capsule(
            capsule_id=capsule_id,
            workspace_id=request.workspace_id,
            created_by=request.created_by,
        )
        assert detail.latest_snapshot.execution_id == execution_id
        assert detail.query_summary == "蓝紫色黄昏"

        updated = await history.set_favorite(
            capsule_id=capsule_id,
            workspace_id=request.workspace_id,
            created_by=request.created_by,
            is_favorite=True,
        )
        assert updated.is_favorite is True
        restored = await history.load_request(
            capsule_id=capsule_id,
            workspace_id=request.workspace_id,
            created_by=request.created_by,
        )
        assert restored.query_text == request.query_text
    finally:
        if capsule_id:
            await history.delete_capsule(
                capsule_id=capsule_id,
                workspace_id=request.workspace_id,
                created_by=request.created_by,
            )
        await database.dispose()
