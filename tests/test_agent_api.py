import pytest
from httpx import ASGITransport, AsyncClient

from capsule.agent.runtime import AgentRuntime
from capsule.api.app import create_app
from capsule.config import Settings


@pytest.mark.asyncio
async def test_agent_api_invokes_runtime_without_database_startup() -> None:
    app = create_app(settings=Settings(), agent_runtime=AgentRuntime())
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/agent/invoke",
                json={
                    "thread_id": "api-thread",
                    "user_id": "api-user",
                    "workspace_id": "api-workspace",
                    "message": "开始创作",
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_api_streams_planning_then_complete_response() -> None:
    app = create_app(settings=Settings(), agent_runtime=AgentRuntime())
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/agent/invoke/stream",
                json={
                    "thread_id": "stream-thread",
                    "user_id": "api-user",
                    "workspace_id": "api-workspace",
                    "message": "开始创作",
                },
            )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: phase\ndata: {\"phase\": \"planning\"" in response.text
    assert 'event: result\ndata: {"thread_id": "stream-thread"' in response.text
    assert '"status": "completed"' in response.text
    assert response.text.endswith("event: done\ndata: {}\n\n")
