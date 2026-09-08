import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings


class FakeRelationGraphService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def build(
        self,
        *,
        workspace_id: str,
        force_understanding: bool = False,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "workspace_id": workspace_id,
                "force_understanding": force_understanding,
            }
        )
        return {
            "workspace_id": workspace_id,
            "asset_count": 0,
            "entity_count": 0,
            "edge_count": 0,
            "assets": [],
            "entities": [],
            "edges": [],
            "entity_edges": [
                {
                    "source_entity_id": "entity_a",
                    "target_entity_id": "entity_b",
                    "relation": "BELONGS_TO",
                    "description": "A 属于 B。",
                }
            ],
        }


@pytest.mark.asyncio
async def test_relation_graph_api_builds_selected_workspace() -> None:
    service = FakeRelationGraphService()
    app = create_app(
        settings=Settings(),
        relation_graph_service=service,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/relation-graphs/build",
                params={
                    "workspace_id": "workspace_real",
                    "force_understanding": "true",
                },
            )

    assert response.status_code == 200
    assert response.json()["workspace_id"] == "workspace_real"
    assert response.json()["entity_edges"] == [
        {
            "source_entity_id": "entity_a",
            "target_entity_id": "entity_b",
            "relation": "BELONGS_TO",
            "description": "A 属于 B。",
        }
    ]
    assert service.calls == [
        {
            "workspace_id": "workspace_real",
            "force_understanding": True,
        }
    ]


@pytest.mark.asyncio
async def test_relation_graph_api_reports_unavailable_service() -> None:
    app = create_app(settings=Settings(), asset_repository=object())  # type: ignore[arg-type]

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/relation-graphs/build",
                params={"workspace_id": "workspace_real"},
            )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "relation_graph_service_not_ready"
