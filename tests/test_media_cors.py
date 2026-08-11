from fastapi.testclient import TestClient

from capsule.api.app import create_app
from capsule.config import Settings


def test_video_range_headers_are_available_cross_origin() -> None:
    app = create_app(
        settings=Settings(search_cors_origins=["http://localhost:3000"]),
        search_service=object(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        preflight = client.options(
            "/api/v1/assets/asset_video/content?workspace_id=workspace_test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Range",
            },
        )
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "range" in preflight.headers["access-control-allow-headers"].lower()
    assert response.headers["access-control-expose-headers"] == (
        "Accept-Ranges, Content-Range, Content-Length, ETag"
    )
