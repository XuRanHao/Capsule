from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from capsule.api.app import create_app
from capsule.config import Settings
from capsule.enums import (
    ClusterAlgorithm,
    ClusterInternalVariance,
    ClusterRunStatus,
    EmbeddingType,
)
from capsule.schemas import ClusterCapsuleRecord, ClusterRunRecord


class FakeClusterRepository:
    def __init__(self) -> None:
        self.run = ClusterRunRecord(
            cluster_run_id="run_api_test",
            workspace_id="workspace_api_test",
            embedding_type="native_multimodal",
            input_embedding_ids=[],
            dataset_hash="0" * 64,
            sample_count=0,
            preprocessing={"submission_mode": "async_api"},
            parameters={},
            cluster_count=None,
            noise_count=None,
            noise_ratio=None,
            status=ClusterRunStatus.PENDING.value,
            started_at=None,
            completed_at=None,
        )
        self.capsule = ClusterCapsuleRecord(
            cluster_capsule_id="cc_api_test",
            cluster_run_id="run_api_test",
            workspace_id="workspace_api_test",
            embedding_type="native_multimodal",
            cluster_label=0,
            model_generated_name="模型名称",
            user_override_name=None,
            effective_name="模型名称",
            model_generated_description="模型生成的测试描述。",
            user_override_description=None,
            effective_description="模型生成的测试描述。",
            keywords=["测试", "聚类", "素材"],
            common_features=["测试特征"],
            internal_variance=ClusterInternalVariance.LOW,
            member_count=15,
            average_membership_probability=0.8,
            medoid_asset_id="asset_medoid",
            representative_asset_ids=["asset_medoid"],
            is_favorite=False,
        )

    async def create_pending_run(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        preprocessing: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> str:
        if workspace_id != self.run.workspace_id:
            raise ValueError("workspace does not exist")
        self.run.embedding_type = embedding_type
        self.run.preprocessing = {
            "submission_mode": "async_api",
            **(preprocessing or {}),
        }
        self.run.parameters = dict(parameters or {})
        return self.run.cluster_run_id

    async def get_run(self, *, cluster_run_id: str, workspace_id: str) -> ClusterRunRecord:
        if cluster_run_id != self.run.cluster_run_id or workspace_id != self.run.workspace_id:
            raise ValueError("cluster run does not exist")
        return self.run

    async def list_capsules(
        self,
        *,
        cluster_run_id: str,
        workspace_id: str,
    ) -> list[ClusterCapsuleRecord]:
        await self.get_run(cluster_run_id=cluster_run_id, workspace_id=workspace_id)
        return [self.capsule]

    async def update_overrides(self, **values: object) -> ClusterCapsuleRecord:
        if values["cluster_capsule_id"] != self.capsule.cluster_capsule_id:
            raise ValueError("cluster capsule does not exist")
        if bool(values["update_name"]):
            name = values["name"]
            assert name is None or isinstance(name, str)
            self.capsule.user_override_name = name
            self.capsule.effective_name = name or self.capsule.model_generated_name
        if bool(values["update_description"]):
            description = values["description"]
            assert description is None or isinstance(description, str)
            self.capsule.user_override_description = description
            self.capsule.effective_description = (
                description or self.capsule.model_generated_description
            )
        return self.capsule


class FakeClusterService:
    def __init__(self, repository: FakeClusterRepository) -> None:
        self.repository = repository
        self.calls: list[dict[str, object]] = []

    async def run(self, **values: object) -> SimpleNamespace:
        self.calls.append(values)
        self.repository.run.status = ClusterRunStatus.COMPLETED.value
        return SimpleNamespace(status=ClusterRunStatus.COMPLETED)


class FakeRelationGraphRebuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def build(self, **values: object) -> dict[str, object]:
        self.calls.append(values)
        return {}


@pytest.mark.asyncio
async def test_cluster_api_submits_one_default_type_and_exposes_polling_routes() -> None:
    repository = FakeClusterRepository()
    service = FakeClusterService(repository)
    app = create_app(
        settings=Settings(),
        cluster_service=service,  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
    )
    transport = ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            submitted = await client.post(
                "/api/v1/cluster-runs",
                json={"workspace_id": "workspace_api_test"},
            )
            polled = await client.get(
                "/api/v1/cluster-runs/run_api_test",
                params={"workspace_id": "workspace_api_test"},
            )
            capsules = await client.get(
                "/api/v1/cluster-runs/run_api_test/capsules",
                params={"workspace_id": "workspace_api_test"},
            )
            patched = await client.patch(
                "/api/v1/cluster-capsules/cc_api_test",
                params={"workspace_id": "workspace_api_test"},
                json={"name": "人工名称", "description": "人工说明"},
            )

    assert submitted.status_code == 202
    assert submitted.json() == {"cluster_run_id": "run_api_test", "status": "pending"}
    assert repository.run.preprocessing["requested_pca_dimension"] == 8
    assert repository.run.preprocessing["vector_fusion"] == {
        "requested_native_content_weight": 0.3,
        "effective_native_content_weight": 1.0,
        "native_content_weight": 1.0,
        "dimension_weight": 0.0,
    }
    assert repository.run.parameters["algorithm"] == "complete_link"
    assert repository.run.parameters["distance_threshold"] == 0.5
    assert repository.run.parameters["min_cluster_size"] == 2
    assert service.calls == [
        {
            "workspace_id": "workspace_api_test",
            "embedding_type": EmbeddingType.NATIVE_MULTIMODAL,
            "cluster_run_id": "run_api_test",
            "pca_dimension": 8,
            "min_samples": 3,
            "min_cluster_size": 2,
            "optimize_parameters": False,
            "algorithm": ClusterAlgorithm.COMPLETE_LINK,
            "distance_threshold": 0.5,
            "native_content_weight": 0.3,
        }
    ]
    assert polled.json()["status"] == "completed"
    assert capsules.json()["items"][0]["medoid_asset_id"] == "asset_medoid"
    assert patched.json()["effective_name"] == "人工名称"
    assert patched.json()["effective_description"] == "人工说明"


@pytest.mark.asyncio
async def test_cluster_api_forwards_optional_parameter_optimization() -> None:
    repository = FakeClusterRepository()
    service = FakeClusterService(repository)
    app = create_app(
        settings=Settings(),
        cluster_service=service,  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/cluster-runs",
                json={
                    "workspace_id": "workspace_api_test",
                    "embedding_type": "visual_presentation",
                    "algorithm": "hdbscan",
                    "pca_dimension": 12,
                    "min_samples": 2,
                    "min_cluster_size": 5,
                    "optimize_parameters": True,
                    "native_content_weight": 0.25,
                },
            )

    assert response.status_code == 202
    assert repository.run.preprocessing["requested_pca_dimension"] == 12
    assert repository.run.preprocessing["vector_fusion"] == {
        "requested_native_content_weight": 0.25,
        "effective_native_content_weight": 0.25,
        "native_content_weight": 0.25,
        "dimension_weight": 0.75,
    }
    assert repository.run.parameters["min_samples"] == 2
    assert repository.run.parameters["min_cluster_size"] == 5
    assert service.calls == [
        {
            "workspace_id": "workspace_api_test",
            "embedding_type": EmbeddingType.VISUAL_PRESENTATION,
            "cluster_run_id": "run_api_test",
            "pca_dimension": 12,
            "min_samples": 2,
            "min_cluster_size": 5,
            "optimize_parameters": True,
            "algorithm": ClusterAlgorithm.HDBSCAN,
            "distance_threshold": 0.5,
            "native_content_weight": 0.25,
        }
    ]


@pytest.mark.asyncio
async def test_cluster_api_forwards_complete_link_parameters() -> None:
    repository = FakeClusterRepository()
    service = FakeClusterService(repository)
    app = create_app(
        settings=Settings(),
        cluster_service=service,  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/cluster-runs",
                json={
                    "workspace_id": "workspace_api_test",
                    "embedding_type": "visual_presentation",
                    "algorithm": "complete_link",
                    "pca_dimension": 8,
                    "distance_threshold": 0.35,
                    "min_cluster_size": 4,
                },
            )

    assert response.status_code == 202
    assert repository.run.parameters == {
        "algorithm": "complete_link",
        "distance_threshold": 0.35,
        "min_cluster_size": 4,
        "metric": "euclidean",
        "linkage": "complete",
    }
    assert service.calls[0]["algorithm"] is ClusterAlgorithm.COMPLETE_LINK
    assert service.calls[0]["distance_threshold"] == 0.35


@pytest.mark.asyncio
@pytest.mark.parametrize("native_content_weight", [0.0, 1.0])
async def test_cluster_api_accepts_fusion_weight_endpoints(native_content_weight: float) -> None:
    repository = FakeClusterRepository()
    service = FakeClusterService(repository)
    app = create_app(
        settings=Settings(),
        cluster_service=service,  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/cluster-runs",
                json={
                    "workspace_id": "workspace_api_test",
                    "embedding_type": "visual_presentation",
                    "native_content_weight": native_content_weight,
                },
            )

    assert response.status_code == 202
    assert service.calls[0]["native_content_weight"] == native_content_weight


@pytest.mark.asyncio
async def test_user_subject_clustering_rebuilds_relation_graph_after_completion() -> None:
    repository = FakeClusterRepository()
    service = FakeClusterService(repository)
    rebuilder = FakeRelationGraphRebuilder()
    app = create_app(
        settings=Settings(),
        cluster_service=service,  # type: ignore[arg-type]
        cluster_repository=repository,  # type: ignore[arg-type]
        relation_graph_service=rebuilder,  # type: ignore[arg-type]
    )

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/cluster-runs",
                json={
                    "workspace_id": "workspace_api_test",
                    "embedding_type": "subject_content",
                },
            )

    assert response.status_code == 202
    assert rebuilder.calls == [
        {"workspace_id": "workspace_api_test", "force_rebuild": True}
    ]
