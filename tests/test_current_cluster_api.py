from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.db.repositories import (
    CurrentClusterMemberRecord as PersistedCurrentClusterMemberRecord,
)
from capsule.db.repositories import CurrentClusterRecord as PersistedCurrentClusterRecord
from capsule.db.repositories import (
    NewAssetClusterStatusItemRecord,
    NewAssetClusterStatusRecord,
)
from capsule.enums import (
    AssetType,
    ClusterMemberSource,
    ClusterMode,
    NewAssetClusterStatus,
)
from capsule.schemas import (
    CurrentClusterListResponse,
    CurrentClusterMemberListResponse,
    CurrentClusterMemberRecord,
    CurrentClusterRecord,
)


def test_current_cluster_responses_accept_repository_records() -> None:
    now = datetime(2026, 8, 5, tzinfo=UTC)
    cluster = PersistedCurrentClusterRecord(
        cluster_id="cluster_persisted",
        workspace_id="workspace_persisted",
        embedding_type="subject_content",
        mode=ClusterMode.DYNAMIC,
        name="持久化簇",
        description="来自 repository dataclass",
        representative_asset_id="asset_1",
        source_run_id="run_1",
        created_at=now,
        updated_at=now,
    )
    member = PersistedCurrentClusterMemberRecord(
        cluster_id=cluster.cluster_id,
        asset_id="asset_1",
        embedding_type=cluster.embedding_type,
        source=ClusterMemberSource.FULL_CLUSTER,
        score=0.97,
        created_at=now,
    )

    cluster_response = CurrentClusterListResponse(items=[cluster])
    member_response = CurrentClusterMemberListResponse(items=[member])

    assert cluster_response.items[0].cluster_id == cluster.cluster_id
    assert member_response.items[0].asset_id == member.asset_id


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
        self.status_calls: list[dict[str, object]] = []

    async def get_new_asset_cluster_status(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
    ) -> NewAssetClusterStatusRecord:
        self.status_calls.append(
            {
                "workspace_id": workspace_id,
                "embedding_type": embedding_type,
                "model_name": model_name,
                "dimension": dimension,
                "milvus_collection": milvus_collection,
            }
        )
        now = datetime(2026, 8, 5, tzinfo=UTC)
        return NewAssetClusterStatusRecord(
            has_baseline=True,
            baseline_cluster_run_id="run_baseline",
            baseline_sample_count=12,
            eligible_asset_count=15,
            items=(
                NewAssetClusterStatusItemRecord(
                    asset_id="asset_incremental",
                    asset_type=AssetType.IMAGE,
                    file_name="incremental.png",
                    asset_name="增量素材",
                    status=NewAssetClusterStatus.INCREMENTALLY_CLUSTERED,
                    cluster_id="cluster_dynamic",
                    cluster_name="动态簇",
                    cluster_mode=ClusterMode.DYNAMIC,
                    member_source=ClusterMemberSource.INCREMENTAL,
                    score=0.91,
                    created_at=now,
                ),
                NewAssetClusterStatusItemRecord(
                    asset_id="asset_pending",
                    asset_type=AssetType.TEXT_BLOCK,
                    file_name="pending.md",
                    asset_name=None,
                    status=NewAssetClusterStatus.PENDING,
                    cluster_id=None,
                    cluster_name=None,
                    cluster_mode=None,
                    member_source=None,
                    score=None,
                    created_at=now,
                ),
                NewAssetClusterStatusItemRecord(
                    asset_id="asset_manual",
                    asset_type=AssetType.VIDEO_SEGMENT,
                    file_name="manual.mp4",
                    asset_name="人工管理素材",
                    status=NewAssetClusterStatus.MANUAL_MANAGEMENT,
                    cluster_id="cluster_manual",
                    cluster_name="人工簇",
                    cluster_mode=ClusterMode.RESIDENT_MANUAL,
                    member_source=ClusterMemberSource.USER,
                    score=None,
                    created_at=now,
                ),
            ),
        )

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

    async def set_name(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        name: str,
    ) -> CurrentClusterRecord:
        await self.get_cluster(cluster_id=cluster_id, workspace_id=workspace_id)
        self.cluster = self.cluster.model_copy(update={"name": name})
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
    app = create_app(  # type: ignore[arg-type]
        settings=Settings(_env_file=None),
        cluster_repository=object(),
    )
    app.state.current_cluster_repository = repository
    return app


@pytest.mark.asyncio
async def test_new_asset_cluster_status_api_reports_baseline_and_three_states() -> None:
    repository = FakeCurrentClusterRepository()
    app = _app_with_repository(repository)

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/v1/clusters/assets/status",
                params={
                    "workspace_id": "workspace_current_test",
                    "embedding_type": "visual_style",
                },
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"] == "workspace_current_test"
    assert payload["embedding_type"] == "visual_style"
    assert payload["initialized"] is True
    assert payload["bootstrap_minimum_count"] == 50
    assert payload["baseline_cluster_run_id"] == "run_baseline"
    assert payload["baseline_sample_count"] == 12
    assert payload["eligible_asset_count"] == 15
    assert payload["new_asset_count"] == 3
    assert payload["incrementally_clustered_count"] == 1
    assert payload["pending_count"] == 1
    assert payload["manual_management_count"] == 1
    assert {item["status"] for item in payload["items"]} == {
        "incrementally_clustered",
        "pending",
        "manual_management",
    }
    assert repository.status_calls == [
        {
            "workspace_id": "workspace_current_test",
            "embedding_type": "visual_style",
            "model_name": "doubao-embedding-vision-250615",
            "dimension": 1024,
            "milvus_collection": "asset_embeddings_seed16_1024",
        }
    ]


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
            renamed = await client.patch(
                "/api/v1/clusters/cluster_current_test",
                params={"workspace_id": "workspace_current_test"},
                json={"name": "  新的簇名  "},
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
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "新的簇名"
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
