import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from capsule.db.repositories import EmbeddingAsset
from capsule.enums import ClusterRunStatus, EmbeddingType
from capsule.pipeline.cluster_service import EmbeddingTypeClusterResult
from capsule.pipeline.relation_graph_service import RelationGraphService
from capsule.relation_graph import (
    AssetEntityRelationResolution,
    EntityStructureOperationResolution,
    MergedEntityResolution,
    MetadataContentResolution,
)
from capsule.schemas import AssetUnderstanding, EmbeddingResult
from capsule.search.models import VectorSearchHit


def _understanding(subject: str, description: str) -> AssetUnderstanding:
    return AssetUnderstanding.model_validate(
        {
            "asset_name": subject,
            "asset_description": description,
            "features": {
                "subject_content": {
                    "applicability": "applicable",
                    "items": [
                        {
                            "subject": subject,
                            "description": description,
                            "salience": 0.9,
                            "status": "observed",
                            "evidence": [],
                            "ocr_confidence": None,
                        }
                    ],
                }
            },
        }
    )


def _asset(asset_id: str, path: str, understanding: AssetUnderstanding) -> EmbeddingAsset:
    return EmbeddingAsset(
        asset_id=asset_id,
        workspace_id="workspace_real",
        project_id="project_real",
        source_file_id=f"source_{asset_id}",
        asset_type="image",
        file_type=".png",
        content_hash=asset_id * 8,
        embedding_revision=1,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        raw_content=None,
        asset_description=understanding.asset_description,
        asset_features=understanding.features.model_dump(mode="json"),
        derived_file_uri=None,
        source_storage_uri="file:///tmp/test.png",
        source_mime_type="image/png",
        file_name=path.rsplit("/", 1)[-1],
        source_relative_path=path,
    )


class FakeEmbeddingRepository:
    def __init__(self, assets: list[EmbeddingAsset]) -> None:
        self.assets = assets

    async def list_assets(self, *, workspace_id: str) -> list[EmbeddingAsset]:
        assert workspace_id == "workspace_real"
        return self.assets


class UnexpectedUnderstandingService:
    async def run(self, **_: object) -> object:
        raise AssertionError("stored understanding should be reused")


class GeneratedSubjectCluster:
    cluster_id = "cluster_subject"
    name = "飞船大厅场景"
    description = "飞船内部的大厅空间"
    embedding_vector = (1.0, 0.0)
    embedding_model = "stored-embedding"
    embedding_source_hash = "stored-hash"


class GeneratedSubjectMember:
    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id


class EmptyThenGeneratedClusterRepository:
    def __init__(self) -> None:
        self.generated = False

    async def list_clusters(self, **_: object) -> list[GeneratedSubjectCluster]:
        return [GeneratedSubjectCluster()] if self.generated else []

    async def list_members(self, **_: object) -> list[GeneratedSubjectMember]:
        return [GeneratedSubjectMember("asset_a"), GeneratedSubjectMember("asset_b")]


class SubjectClusterRunner:
    def __init__(self, repository: EmptyThenGeneratedClusterRepository) -> None:
        self.repository = repository
        self.calls = 0

    async def run(self, **kwargs: object) -> EmbeddingTypeClusterResult:
        assert kwargs["embedding_type"] is EmbeddingType.SUBJECT_CONTENT
        assert kwargs["trigger"] == "relation_graph"
        self.calls += 1
        self.repository.generated = True
        return EmbeddingTypeClusterResult(
            embedding_type=EmbeddingType.SUBJECT_CONTENT,
            cluster_run_id="run_subject",
            status=ClusterRunStatus.COMPLETED,
            indexed_asset_count=2,
            vector_count=2,
            cluster_count=1,
        )


class FakeRelationModel:
    def __init__(self) -> None:
        self.edge_candidates: list[dict[str, object]] = []
        self.merge_calls = 0
        self.embed_calls = 0

    async def resolve_metadata_content_entities(
        self,
        groups: list[dict[str, object]],
        *,
        guidance: str | None = None,
    ) -> MetadataContentResolution:
        assert guidance is None
        assert groups[0]["metadata_entity"] == "飞船内部"
        return MetadataContentResolution.model_validate(
            {
                "decisions": [
                    {
                        "metadata_entity": "飞船内部",
                        "relation": "contains_content",
                        "description": "目录表达共同场景。",
                        "build_entity": True,
                    }
                ]
            }
        )

    async def generate_asset_entity_relations(
        self,
        candidates: list[dict[str, object]],
    ) -> AssetEntityRelationResolution:
        self.edge_candidates = candidates
        return AssetEntityRelationResolution.model_validate(
            {
                "relations": [
                    {
                        "source_id": item["source_id"],
                        "target_id": item["target_id"],
                        "establishes_relation": True,
                        "relation": "SET_IN",
                        "description": "该素材的画面位于飞船内部。",
                    }
                    for item in candidates
                ]
            }
        )

    async def merge_entity_candidates(
        self,
        candidates: list[dict[str, object]],
    ) -> MergedEntityResolution:
        self.merge_calls += 1
        return MergedEntityResolution.model_validate(
            {
                "entities": [
                    {
                        "name": item["name"],
                        "semantic": item["semantic"],
                        "candidate_ids": [item["candidate_id"]],
                        "build_entity": True,
                        "reason": "候选具有实际场景语义。",
                    }
                    for item in candidates
                ]
            }
        )

    async def embed_text(self, text: str) -> EmbeddingResult:
        self.embed_calls += 1
        return EmbeddingResult(vector=[1.0, 0.0], model="fake-embedding")


class FakeRelationVectorStore:
    async def search_raw(self, **_: object) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                embedding_id="emb_asset_c",
                asset_id="asset_c",
                source_file_id="source_asset_c",
                asset_type="image",
                embedding_revision=1,
                similarity=0.91,
            )
        ]

    async def fetch_vectors(self, _: object) -> dict[str, list[float]]:
        return {}


class SeparatingStructureModel(FakeRelationModel):
    def __init__(self) -> None:
        super().__init__()
        self.structure_calls: list[tuple[dict[str, Any], list[dict[str, Any]], str]] = []

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: dict[str, Any],
        incoming_entities: list[dict[str, Any]],
        workspace_tree: str,
    ) -> EntityStructureOperationResolution:
        self.structure_calls.append((current_graph, incoming_entities, workspace_tree))
        return EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "separate",
                        "entity_ids": [item["entity_id"] for item in incoming_entities],
                    }
                ]
            }
        )


class ConcurrentAssetAndEntityModel(FakeRelationModel):
    def __init__(self) -> None:
        super().__init__()
        self.asset_started = asyncio.Event()
        self.structure_started = asyncio.Event()

    async def generate_asset_entity_relations(
        self,
        candidates: list[dict[str, object]],
    ) -> AssetEntityRelationResolution:
        self.asset_started.set()
        await asyncio.wait_for(self.structure_started.wait(), timeout=1)
        return await super().generate_asset_entity_relations(candidates)

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: dict[str, object],
        incoming_entities: list[dict[str, object]],
        workspace_tree: str,
    ) -> EntityStructureOperationResolution:
        del current_graph, workspace_tree
        self.structure_started.set()
        await asyncio.wait_for(self.asset_started.wait(), timeout=1)
        return EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "separate",
                        "entity_ids": [item["entity_id"] for item in incoming_entities],
                    }
                ]
            }
        )


@pytest.mark.asyncio
async def test_asset_judgment_and_entity_structure_run_concurrently() -> None:
    first = _understanding("中央雕塑", "飞船大厅里的大型人面雕塑")
    second = _understanding("宴会大厅", "飞船内部的宴会空间")
    model = ConcurrentAssetAndEntityModel()
    service = RelationGraphService(
        embedding_repository=FakeEmbeddingRepository(
            [
                _asset("asset_a", "飞船内部/中央大厅.png", first),
                _asset("asset_b", "飞船内部/宴会大厅.png", second),
            ]
        ),  # type: ignore[arg-type]
        understanding_service=UnexpectedUnderstandingService(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
        metadata_candidates_enabled=True,
    )

    graph = await service.build(workspace_id="workspace_real")

    assert model.asset_started.is_set()
    assert model.structure_started.is_set()
    assert graph["entity_count"] == 1


@pytest.mark.asyncio
async def test_entity_structure_uses_fifteen_item_incremental_batches() -> None:
    model = SeparatingStructureModel()
    service = RelationGraphService(
        embedding_repository=object(),  # type: ignore[arg-type]
        understanding_service=object(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
    )
    graph: dict[str, Any] = {
        "entities": [
            {
                "entity_id": f"entity_{index}",
                "name": f"实体{index}",
                "semantic": f"实体{index}的语义",
            }
            for index in range(31)
        ],
        "edges": [],
        "entity_edges": [],
    }

    result = await service._structure_entities(graph=graph, assets=[])

    assert [len(call[1]) for call in model.structure_calls] == [15, 15, 1]
    assert [len(call[0]["nodes"]) for call in model.structure_calls] == [0, 15, 30]
    assert all(call[2].startswith("./ [0 Assets]") for call in model.structure_calls)
    assert result == {
        "status": "completed",
        "batch_size": 15,
        "call_count": 3,
        "virtual_round_count": 0,
    }


@pytest.mark.asyncio
async def test_service_builds_live_graph_and_generates_membership_descriptions() -> None:
    first = _understanding("中央雕塑", "飞船大厅里的大型人面雕塑")
    second = _understanding("宴会大厅", "飞船内部的宴会空间")
    repository = FakeEmbeddingRepository(
        [
            _asset("asset_a", "飞船内部/中央大厅.png", first),
            _asset("asset_b", "飞船内部/宴会大厅.png", second),
        ]
    )
    model = FakeRelationModel()
    service = RelationGraphService(
        embedding_repository=repository,  # type: ignore[arg-type]
        understanding_service=UnexpectedUnderstandingService(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
        metadata_candidates_enabled=True,
    )

    graph = await service.build(workspace_id="workspace_real")

    assert graph["workspace_id"] == "workspace_real"
    assert model.merge_calls == 0
    assert graph["entity_count"] == 1
    assert len(model.edge_candidates) == 2
    membership_edges = [
        edge for edge in graph["edges"] if str(edge["target"]).startswith("entity_")
    ]
    assert {edge["relation"] for edge in membership_edges} == {"SET_IN"}
    assert {edge["description"] for edge in membership_edges} == {"该素材的画面位于飞船内部。"}
    assert {edge["content_subject"] for edge in membership_edges} == {
        "中央雕塑",
        "宴会大厅",
    }


@pytest.mark.asyncio
async def test_service_generates_subject_clusters_when_none_exist() -> None:
    first = _understanding("中央雕塑", "飞船大厅里的大型人面雕塑")
    second = _understanding("宴会大厅", "飞船内部的宴会空间")
    embedding_repository = FakeEmbeddingRepository(
        [
            _asset("asset_a", "飞船内部/中央大厅.png", first),
            _asset("asset_b", "飞船内部/宴会大厅.png", second),
        ]
    )
    cluster_repository = EmptyThenGeneratedClusterRepository()
    runner = SubjectClusterRunner(cluster_repository)
    model = FakeRelationModel()
    service = RelationGraphService(
        embedding_repository=embedding_repository,  # type: ignore[arg-type]
        understanding_service=UnexpectedUnderstandingService(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
        current_cluster_repository=cluster_repository,  # type: ignore[arg-type]
        subject_cluster_runner=runner,
    )

    graph = await service.build(workspace_id="workspace_real")

    assert runner.calls == 1
    assert graph["subject_cluster_status"] == {
        "status": "generated",
        "cluster_count": 1,
        "generated": True,
        "cluster_run_id": "run_subject",
        "message": None,
    }
    assert any(candidate["origin"] == "subject_cluster" for candidate in graph["entity_candidates"])
    assert {candidate["origin"] for candidate in graph["entity_candidates"]} == {"subject_cluster"}
    assert model.embed_calls == 0
    assert model.edge_candidates == []
    assert {edge["relation"] for edge in graph["edges"]} == {"CLUSTER_MEMBER"}
    assert {edge["description"] for edge in graph["edges"]} == {"内容高度相似"}


@pytest.mark.asyncio
async def test_service_recalls_assets_outside_original_subject_cluster() -> None:
    first = _understanding("汪叹之", "金发蓝眼角色全身像")
    second = _understanding("汪叹之", "金发蓝眼角色半身像")
    missing = _understanding("金发男性", "金发蓝眼男性面部近景")
    embedding_repository = FakeEmbeddingRepository(
        [
            _asset("asset_a", "汪叹之/全身.png", first),
            _asset("asset_b", "汪叹之/半身.png", second),
            _asset("asset_c", "汪叹之/面部近景.png", missing),
        ]
    )
    cluster_repository = EmptyThenGeneratedClusterRepository()
    cluster_repository.generated = True
    model = FakeRelationModel()
    service = RelationGraphService(
        embedding_repository=embedding_repository,  # type: ignore[arg-type]
        understanding_service=UnexpectedUnderstandingService(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
        current_cluster_repository=cluster_repository,  # type: ignore[arg-type]
        vector_store=FakeRelationVectorStore(),
    )

    graph = await service.build(workspace_id="workspace_real")

    assert graph["recalled_edge_candidate_count"] == 1
    assert len(model.edge_candidates) == 1
    recalled = next(
        candidate for candidate in model.edge_candidates if candidate["source_id"] == "asset_c"
    )
    assert "recall_similarity" not in recalled
    assert "recall_confidence" not in recalled
    assert "path_affinity" not in recalled
    assert graph["entity_count"] == 1
    assert graph["entities"][0]["asset_ids"] == ["asset_a", "asset_b", "asset_c"]
    assert graph["entities"][0]["embedding_model"] == "stored-embedding"
    cluster_edges = [edge for edge in graph["edges"] if edge["source"] in {"asset_a", "asset_b"}]
    assert {edge["relation"] for edge in cluster_edges} == {"CLUSTER_MEMBER"}
    assert {edge["description"] for edge in cluster_edges} == {"内容高度相似"}


class IncrementalSubjectClusterRepository:
    def __init__(self, *, member_ids: list[str]) -> None:
        self.member_ids = member_ids

    async def list_clusters(self, **_: object) -> list[GeneratedSubjectCluster]:
        return [GeneratedSubjectCluster()]

    async def list_members(self, **_: object) -> list[GeneratedSubjectMember]:
        return [GeneratedSubjectMember(asset_id) for asset_id in self.member_ids]

    async def list_indexed_asset_embeddings(self, **_: object) -> list[object]:
        return [SimpleNamespace(asset_id="asset_new", embedding_id="emb_asset_new")]


class IncrementalEntityVectorStore:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.fetch_calls: list[list[str]] = []

    async def fetch_vectors(self, embedding_ids: list[str]) -> dict[str, list[float]]:
        self.fetch_calls.append(embedding_ids)
        return {"emb_asset_new": self.vector}


class IncrementalRelationRepository:
    def __init__(self) -> None:
        self.update_call: dict[str, Any] | None = None

    async def load_current(self, **_: object) -> dict[str, Any]:
        return {
            "entities": [
                {
                    "entity_id": "entity_ship",
                    "name": "飞船大厅场景",
                    "semantic": "飞船内部的大厅空间",
                    "candidate_ids": ["subject_cluster:cluster_subject"],
                    "embedding_vector": [1.0, 0.0],
                },
                {
                    "entity_id": "entity_woody",
                    "name": "伍迪角色素材",
                    "semantic": "伍迪角色的外观素材",
                    "candidate_ids": ["subject_cluster:cluster_woody"],
                    "embedding_vector": [0.0, 1.0],
                },
                {
                    "entity_id": "entity_virtual",
                    "name": "场景",
                    "semantic": "场景类父节点",
                    "candidate_ids": [],
                    "embedding_vector": [],
                },
            ],
            "edges": [],
            "rejected_relations": [],
            "asset_revisions": {},
        }

    async def update_assets(self, **values: Any) -> int:
        self.update_call = values
        return 2


def _incremental_service(
    *,
    cluster_repository: IncrementalSubjectClusterRepository,
    relation_repository: IncrementalRelationRepository,
    vector_store: IncrementalEntityVectorStore,
    model: FakeRelationModel,
    threshold: float = 0.8,
) -> RelationGraphService:
    understanding = _understanding("飞船大厅", "飞船内部的大厅画面")
    return RelationGraphService(
        embedding_repository=FakeEmbeddingRepository(
            [_asset("asset_new", "飞船内部/新大厅.png", understanding)]
        ),  # type: ignore[arg-type]
        understanding_service=UnexpectedUnderstandingService(),  # type: ignore[arg-type]
        model_client=model,  # type: ignore[arg-type]
        current_cluster_repository=cluster_repository,  # type: ignore[arg-type]
        relation_repository=relation_repository,  # type: ignore[arg-type]
        vector_store=vector_store,  # type: ignore[arg-type]
        incremental_entity_recall_similarity_threshold=threshold,
        incremental_entity_recall_top_k=1,
    )


@pytest.mark.asyncio
async def test_incremental_cluster_member_links_directly_without_agent() -> None:
    clusters = IncrementalSubjectClusterRepository(member_ids=["asset_new"])
    relations = IncrementalRelationRepository()
    vectors = IncrementalEntityVectorStore([1.0, 0.0])
    model = FakeRelationModel()
    service = _incremental_service(
        cluster_repository=clusters,
        relation_repository=relations,
        vector_store=vectors,
        model=model,
    )

    result = await service.update_assets(
        workspace_id="workspace_real",
        asset_ids=["asset_new"],
        affected_cluster_ids=["cluster_subject"],
    )

    assert result["cluster_linked_asset_count"] == 1
    assert result["vector_recalled_candidate_count"] == 0
    assert vectors.fetch_calls == []
    assert model.edge_candidates == []
    assert relations.update_call is not None
    resolution = relations.update_call["resolution"]
    assert resolution.relations[0].target_id == "entity_ship"
    assert resolution.relations[0].relation == "CLUSTER_MEMBER"
    assert resolution.relations[0].description == "内容高度相似"


@pytest.mark.asyncio
async def test_incremental_unclustered_asset_recalls_high_similarity_entity_for_agent() -> None:
    clusters = IncrementalSubjectClusterRepository(member_ids=[])
    relations = IncrementalRelationRepository()
    vectors = IncrementalEntityVectorStore([0.99, 0.01])
    model = FakeRelationModel()
    service = _incremental_service(
        cluster_repository=clusters,
        relation_repository=relations,
        vector_store=vectors,
        model=model,
    )

    result = await service.update_assets(
        workspace_id="workspace_real",
        asset_ids=["asset_new"],
        affected_cluster_ids=[],
    )

    assert result["cluster_linked_asset_count"] == 0
    assert result["vector_recalled_asset_count"] == 1
    assert result["vector_recalled_candidate_count"] == 1
    assert [candidate["target_id"] for candidate in model.edge_candidates] == [
        "entity_ship"
    ]
    assert "similarity" not in model.edge_candidates[0]
    assert "recall_similarity" not in model.edge_candidates[0]
    assert relations.update_call is not None
    resolution = relations.update_call["resolution"]
    assert resolution.relations[0].relation == "SET_IN"


@pytest.mark.asyncio
async def test_incremental_unclustered_asset_below_threshold_stays_unlinked() -> None:
    clusters = IncrementalSubjectClusterRepository(member_ids=[])
    relations = IncrementalRelationRepository()
    vectors = IncrementalEntityVectorStore([0.7, 0.7])
    model = FakeRelationModel()
    service = _incremental_service(
        cluster_repository=clusters,
        relation_repository=relations,
        vector_store=vectors,
        model=model,
        threshold=0.9,
    )

    result = await service.update_assets(
        workspace_id="workspace_real",
        asset_ids=["asset_new"],
        affected_cluster_ids=[],
    )

    assert result["vector_recalled_asset_count"] == 0
    assert result["vector_recalled_candidate_count"] == 0
    assert model.edge_candidates == []
    assert relations.update_call is not None
    assert relations.update_call["resolution"].relations == []
