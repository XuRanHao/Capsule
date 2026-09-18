import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings


class RecordingGraphRepository:
    def __init__(self) -> None:
        self.requests: list[dict[str, str]] = []

    async def create_graph(
        self,
        *,
        workspace_id: str,
        name: str,
        description: str,
    ) -> dict[str, str]:
        self.requests.append(
            {
                "workspace_id": workspace_id,
                "name": name,
                "description": description,
            }
        )
        return {
            "graph_id": "graph-created",
            "workspace_id": workspace_id,
            "name": name,
            "description": description,
        }


@pytest.mark.asyncio
async def test_create_narrative_graph_returns_the_real_graph_identifier() -> None:
    repository = RecordingGraphRepository()
    app = create_app(
        settings=Settings(),
        graph_repository=repository,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/graphs",
                json={"workspace_id": "workspace-a"},
            )

    assert response.status_code == 201
    assert response.json() == {
        "graph_id": "graph-created",
        "workspace_id": "workspace-a",
        "name": "未命名图谱",
        "description": "",
    }
    assert repository.requests == [
        {
            "workspace_id": "workspace-a",
            "name": "未命名图谱",
            "description": "",
        }
    ]
