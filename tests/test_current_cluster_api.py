from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.enums import ClusterMemberSource, ClusterMode
from capsule.schemas import CurrentClusterMemberRecord, CurrentClusterRecord


class FakeCurrentClusterRepository:
    def __init__(self) -> None:
        now = datetime(2026, 8, 4, tzinfo=UTC)
        self.cluster = CurrentClusterRecord(
            cluster_id="cluster_current_test",
            workspace_id="workspace_current_test",
            embedding_type="visual_style",
            mode=ClusterMode.DYNAMIC,
            name="极简视觉",
            description="极简视觉风格素材",
            representative_asset_id="asset_representative",
            source_run_id="run_current_test",
            created_at=now,
            updated_at=now,
        )
        self.members: dict[str, CurrentClusterMemberRecord] = {}
        self.exclusions: set[tuple[str, str]] = set()
        self.attach_calls: list[list[str]] = []
        self.detach_calls: list[list[str]] = []

    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: list[ClusterMode] | None = None,
    ) -> list[CurrentClusterRecord]:
        if workspace_id != self.cluster.workspace_id:
            return []
        if embedding_type != self.cluster.embedding_type:
            return []
        if modes is not None and self.cluster.mode not in modes:
            return []
        return [self.cluster]

    async def get_cluster(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> CurrentClusterRecord:
        if cluster_id != self.cluster.cluster_id or workspace_id != self.cluster.workspace_id:
            raise ValueError("current cluster does not exist in workspace")
        return self.cluster

    async def list_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> list[CurrentClusterMemberRecord]:
        await self.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        return [item for item in self.members.values() if item.cluster_id == cluster_id]

    async def set_mode(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        mode: ClusterMode,
    ) -> CurrentClusterRecord:
        await self.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        self.cluster = self.cluster.model_copy(update={"mode": mode})
        return self.cluster

    async def attach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: list[str],
        source: ClusterMemberSource = ClusterMemberSource.USER,
        scores: dict[str, float] | None = None,
    ) -> list[CurrentClusterMemberRecord]:
        cluster = await self.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        if cluster.mode == ClusterMode.DYNAMIC:
            raise ValueError("members can only be attached to resident clusters")
        if "asset_without_embedding" in asset_ids:
            raise ValueError("asset does not have an indexed embedding for visual_style")
        self.attach_calls.append(asset_ids)
        now = datetime(2026, 8, 4, tzinfo=UTC)
        attached: list[CurrentClusterMemberRecord] = []
        for asset_id in asset_ids:
            member = CurrentClusterMemberRecord(
                cluster_id=cluster_id,
                asset_id=asset_id,
                embedding_type=cluster.embedding_type,
                source=source,
                score=None if scores is None else scores.get(asset_id),
                created_at=now,
            )
            self.members[asset_id] = member
            self.exclusions.discard((cluster_id, asset_id))
            attached.append(member)
        return attached

    async def detach_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        asset_ids: list[str],
        created_by: str | None = None,
    ) -> list[str]:
        await self.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        self.detach_calls.append(asset_ids)
        detached: list[str] = []
        for asset_id in asset_ids:
            member = self.members.get(asset_id)
            if member is not None and member.cluster_id == cluster_id:
                self.members.pop(asset_id)
                detached.append(asset_id)
            self.exclusions.add((cluster_id, asset_id))
        return detached


def _app_with_repository(repository: FakeCurrentClusterRepository):
    # Supplying any repository keeps the test lifespan from constructing external services.
    app = create_app(settings=Settings(), cluster_repository=object())  # type: ignore[arg-type]
    app.state.current_cluster_repository = repository
    return app


@pytest.mark.asyncio
async def test_current_cluster_api_supports_resident_membership_workflow() -> None:
    repository = FakeCurrentClusterRepository()
    app = _app_with_repository(repository)

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            listed = await client.get(
                "/api/v1/clusters",
                params={
                    "workspace_id": "workspace_current_test",
                    "embedding_type": "visual_style",
                },
            )
            patched = await client.patch(
                "/api/v1/clusters/cluster_current_test",
                params={"workspace_id": "workspace_current_test"},
                json={"mode": "resident_open"},
            )
            attached = await client.post(
                "/api/v1/clusters/cluster_current_test/members:attach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_1", "asset_1", "asset_2"]},
            )
            attached_again = await client.post(
                "/api/v1/clusters/cluster_current_test/members:attach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_1", "asset_2"]},
            )
            members = await client.get(
                "/api/v1/clusters/cluster_current_test/members",
                params={"workspace_id": "workspace_current_test"},
            )
            detached = await client.post(
                "/api/v1/clusters/cluster_current_test/members:detach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_1"]},
            )
            detached_again = await client.post(
                "/api/v1/clusters/cluster_current_test/members:detach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_1"]},
            )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["mode"] == "dynamic"
    assert patched.status_code == 200
    assert patched.json()["mode"] == "resident_open"
    assert attached.json() == {
        "cluster_id": "cluster_current_test",
        "asset_ids": ["asset_1", "asset_2"],
    }
    assert attached_again.status_code == 200
    assert repository.attach_calls == [
        ["asset_1", "asset_2"],
        ["asset_1", "asset_2"],
    ]
    assert {item["asset_id"] for item in members.json()["items"]} == {
        "asset_1",
        "asset_2",
    }
    assert all(item["source"] == "user" for item in members.json()["items"])
    assert detached.json()["asset_ids"] == ["asset_1"]
    assert detached_again.status_code == 200
    assert ("cluster_current_test", "asset_1") in repository.exclusions
    assert "asset_1" not in repository.members


@pytest.mark.asyncio
async def test_current_cluster_api_returns_structured_domain_errors() -> None:
    repository = FakeCurrentClusterRepository()
    app = _app_with_repository(repository)

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            missing = await client.get(
                "/api/v1/clusters/missing/members",
                params={"workspace_id": "workspace_current_test"},
            )
            conflict = await client.post(
                "/api/v1/clusters/cluster_current_test/members:attach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_1"]},
            )
            await client.patch(
                "/api/v1/clusters/cluster_current_test",
                params={"workspace_id": "workspace_current_test"},
                json={"mode": "resident_open"},
            )
            invalid = await client.post(
                "/api/v1/clusters/cluster_current_test/members:attach",
                params={"workspace_id": "workspace_current_test"},
                json={"asset_ids": ["asset_without_embedding"]},
            )

    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "current_cluster_not_found"
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "cluster_membership_conflict"
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "invalid_cluster_membership_change"


@pytest.mark.asyncio
async def test_current_cluster_api_requires_repository_and_valid_dimension() -> None:
    app = create_app(settings=Settings(), cluster_repository=object())  # type: ignore[arg-type]

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            unavailable = await client.get(
                "/api/v1/clusters",
                params={
                    "workspace_id": "workspace_current_test",
                    "embedding_type": "visual_style",
                },
            )
            invalid_dimension = await client.get(
                "/api/v1/clusters",
                params={
                    "workspace_id": "workspace_current_test",
                    "embedding_type": "not_a_dimension",
                },
            )

    assert unavailable.status_code == 503
    assert unavailable.json()["detail"]["code"] == "current_cluster_repository_not_ready"
    assert invalid_dimension.status_code == 422
