"""Build one workspace relationship graph from persisted Asset understanding."""

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from capsule.db.repositories import (
    EmbeddingAsset,
    EmbeddingRepository,
    HierarchyAssetAssignment,
    HierarchyBatchCommit,
    HierarchyCandidateSync,
    HierarchyEntityCandidate,
    HierarchyEntityWrite,
    HierarchyGraphInitializationResult,
)
from capsule.enums import ClusterMode, EmbeddingType
from capsule.pipeline.entity_hierarchy_agent import EntityHierarchyAgentAdapter
from capsule.pipeline.entity_hierarchy_workflow import (
    EntityHierarchyWorkflow,
    HierarchyCommitRequest,
    HierarchyCommitResult,
)
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.pipeline.workspace_tree import render_imported_workspace_tree
from capsule.relation_graph import (
    AssetEntityAssignmentRequest,
    AssetEntityAssignmentResolution,
    EntityStructureNode,
    EntityStructureOperationResolution,
    MergedEntityResolution,
)
from capsule.schemas import AssetUnderstanding, EmbeddingResult


class RelationResolutionClient(Protocol):
    async def assign_assets_to_entities(
        self,
        assignments: Sequence[Mapping[str, Any]],
    ) -> AssetEntityAssignmentResolution: ...

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: Mapping[str, Any],
        incoming_entities: Sequence[Mapping[str, Any]],
        workspace_tree: str,
        repair_guidance: str | None = None,
    ) -> EntityStructureOperationResolution: ...

    async def embed_text(self, text: str) -> EmbeddingResult: ...


class AssetVectorSearch(Protocol):
    async def fetch_vectors(
        self,
        embedding_ids: Sequence[str],
    ) -> dict[str, list[float]]: ...


class SubjectClusterRepository(Protocol):
    async def list_clusters(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        modes: Sequence[ClusterMode] | None = None,
    ) -> Sequence[Any]: ...

    async def list_members(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
    ) -> list[Any]: ...

    async def set_embedding(
        self,
        *,
        cluster_id: str,
        workspace_id: str,
        vector: Sequence[float],
        model: str,
        source_hash: str,
    ) -> None: ...

    async def list_indexed_asset_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        asset_ids: Sequence[str],
    ) -> Sequence[Any]: ...


class RelationPersistenceRepository(Protocol):
    async def get_hierarchy_build_state(
        self,
        *,
        workspace_id: str,
    ) -> Mapping[str, Any] | None: ...

    async def initialize_hierarchy_graph(
        self,
        *,
        workspace_id: str,
        input_revision: str,
        subject_cluster_status: Mapping[str, Any],
        expected_build_version: int | None = None,
        expected_input_revision: str | None = None,
    ) -> HierarchyGraphInitializationResult: ...

    async def sync_hierarchy_entity_candidates(
        self,
        *,
        workspace_id: str,
        sync: HierarchyCandidateSync,
    ) -> Any: ...

    async def load(
        self,
        *,
        workspace_id: str,
        input_revision: str,
    ) -> dict[str, Any] | None: ...

    async def load_current(self, *, workspace_id: str) -> dict[str, Any] | None: ...

    async def load_entity_nodes(
        self,
        *,
        workspace_id: str,
        entity_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    async def search_similar_entity_nodes(
        self,
        *,
        workspace_id: str,
        incoming_entity_ids: Sequence[str],
        exclude_entity_ids: Sequence[str] = (),
        per_entity_limit: int = 3,
    ) -> Sequence[Mapping[str, Any]]: ...

    async def load_complete_entity_trees(
        self,
        *,
        workspace_id: str,
        root_entity_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    async def load_workspace_tree(self, *, workspace_id: str) -> str: ...

    async def commit_hierarchy_batch(
        self,
        *,
        workspace_id: str,
        commit: HierarchyBatchCommit,
    ) -> Any: ...

    async def apply_asset_assignments(
        self,
        *,
        workspace_id: str,
        operation_id: str,
        expected_build_version: int,
        expected_input_revision: str,
        input_revision: str,
        subject_cluster_status: Mapping[str, Any],
        assignments: Sequence[HierarchyAssetAssignment],
        asset_revisions: Mapping[str, str],
    ) -> Any: ...


@dataclass(slots=True)
class _HierarchyRepositoryAdapter:
    """Bridge the LangGraph contract to the relationship-graph repository."""

    repository: Any
    model_client: RelationResolutionClient
    expected_build_version: int
    expected_input_revision: str
    subject_cluster_status: Mapping[str, Any]

    async def load_entity_nodes(
        self,
        *,
        workspace_id: str,
        entity_ids: Sequence[str],
    ) -> Sequence[EntityStructureNode]:
        rows = await self.repository.load_entity_nodes(
            workspace_id=workspace_id,
            entity_ids=entity_ids,
        )
        return [EntityStructureNode.model_validate(row) for row in rows]

    async def search_similar_entity_nodes(
        self,
        *,
        workspace_id: str,
        incoming_entity_ids: Sequence[str],
        exclude_entity_ids: Sequence[str],
        per_entity_limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        return await self.repository.search_similar_entity_nodes(
            workspace_id=workspace_id,
            incoming_entity_ids=incoming_entity_ids,
            exclude_entity_ids=exclude_entity_ids,
            per_entity_limit=per_entity_limit,
        )

    async def load_complete_entity_trees(
        self,
        *,
        workspace_id: str,
        root_entity_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        return await self.repository.load_complete_entity_trees(
            workspace_id=workspace_id,
            root_entity_ids=root_entity_ids,
        )

    async def load_workspace_tree(self, *, workspace_id: str) -> str:
        return await self.repository.load_workspace_tree(workspace_id=workspace_id)

    async def commit_hierarchy_batch(
        self,
        request: HierarchyCommitRequest,
    ) -> HierarchyCommitResult:
        entity_writes = await self._created_parent_writes(request)
        result = await self.repository.commit_hierarchy_batch(
            workspace_id=request.workspace_id,
            commit=HierarchyBatchCommit(
                operation_id=request.operation_id,
                expected_build_version=self.expected_build_version,
                expected_input_revision=self.expected_input_revision,
                input_revision=request.input_revision,
                subject_cluster_status=self.subject_cluster_status,
                operations=request.resolution.model_dump(mode="json")["operations"],
                entity_writes=entity_writes,
            ),
        )
        self.expected_build_version = result.build_version
        self.expected_input_revision = result.input_revision
        return HierarchyCommitResult(
            operation_id=result.operation_id,
            created_parent_entity_ids=list(result.created_entity_ids),
        )

    async def _created_parent_writes(
        self,
        request: HierarchyCommitRequest,
    ) -> list[HierarchyEntityWrite]:
        writes: list[HierarchyEntityWrite] = []
        embed_text = getattr(self.model_client, "embed_text", None)
        if not callable(embed_text):
            raise RuntimeError("entity hierarchy requires an Entity semantic embedding client")
        for operation in request.resolution.operations:
            parent = getattr(operation, "parent", None)
            if getattr(parent, "mode", None) != "create":
                continue
            temporary_id = str(parent.temporary_parent_id)
            entity_id = request.created_parent_ids_by_temporary_id.get(temporary_id)
            if entity_id is None:
                raise ValueError(f"missing persisted ID for {temporary_id}")
            result = await embed_text(f"{parent.name}\n{parent.semantic}")
            vector = _normalize_vector(result.vector)
            if vector is None:
                raise ValueError(f"invalid embedding for new hierarchy parent {temporary_id}")
            writes.append(
                HierarchyEntityWrite(
                    entity_id=entity_id,
                    name=parent.name,
                    semantic=parent.semantic,
                    origins=("hierarchy_group",),
                    descriptions=(parent.semantic,),
                    merge_reason="Entity hierarchy Agent created parent",
                    embedding_vector=vector,
                    embedding_model=result.model,
                    temporary_parent_id=temporary_id,
                )
            )
        return writes


class RelationGraphService:
    def __init__(
        self,
        *,
        embedding_repository: EmbeddingRepository,
        understanding_service: AssetUnderstandingService,
        model_client: RelationResolutionClient,
        current_cluster_repository: SubjectClusterRepository | None = None,
        relation_repository: RelationPersistenceRepository | None = None,
        vector_store: AssetVectorSearch | None = None,
        incremental_entity_recall_similarity_threshold: float = 0.72,
        incremental_entity_recall_top_k: int = 3,
        hierarchy_checkpointer: Any | None = None,
    ) -> None:
        self._embedding_repository = embedding_repository
        self._understanding_service = understanding_service
        self._model_client = model_client
        self._current_cluster_repository = current_cluster_repository
        self._relation_repository = relation_repository
        self._vector_store = vector_store
        self._incremental_entity_recall_similarity_threshold = (
            incremental_entity_recall_similarity_threshold
        )
        self._incremental_entity_recall_top_k = incremental_entity_recall_top_k
        self._hierarchy_checkpointer = hierarchy_checkpointer
        self._build_tasks: dict[tuple[str, bool, str], asyncio.Task[dict[str, Any]]] = {}
        self._build_tasks_lock = asyncio.Lock()

    async def build(
        self,
        *,
        workspace_id: str,
        force_understanding: bool = False,
    ) -> dict[str, Any]:
        assets = await self._embedding_repository.list_assets(workspace_id=workspace_id)
        snapshot = _asset_snapshot(assets)
        key = (workspace_id, force_understanding, snapshot)
        async with self._build_tasks_lock:
            task = self._build_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._build_once(
                        workspace_id=workspace_id,
                        force_understanding=force_understanding,
                        assets=assets,
                    )
                )
                self._build_tasks[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._build_tasks_lock:
                    if self._build_tasks.get(key) is task:
                        self._build_tasks.pop(key, None)

    async def _build_once(
        self,
        *,
        workspace_id: str,
        force_understanding: bool,
        assets: list[EmbeddingAsset],
    ) -> dict[str, Any]:
        if not assets:
            empty_cluster_status = {
                "status": "empty",
                "cluster_count": 0,
                "generated": False,
            }
            return await self._build_hierarchy_graph(
                workspace_id=workspace_id,
                assets=[],
                usable_assets=[],
                understandings={},
                understanding_errors=[],
                subject_clusters=[],
                subject_member_groups=[],
                subject_cluster_status=empty_cluster_status,
                entity_candidates=[],
                candidate_vectors={},
                embedding_model="",
                force_understanding=force_understanding,
                input_revision=_input_revision(
                    [],
                    subject_clusters=[],
                    subject_member_groups=[],
                    candidate_policy=self._candidate_policy,
                ),
            )

        missing_ids = [
            asset.asset_id
            for asset in assets
            if force_understanding or not asset.asset_description or not asset.asset_features
        ]
        understanding_errors: list[dict[str, str]] = []
        if missing_ids:
            result = await self._understanding_service.run(
                workspace_id=workspace_id,
                asset_ids=missing_ids,
                force=force_understanding,
            )
            understanding_errors = result.errors
            assets = await self._embedding_repository.list_assets(workspace_id=workspace_id)

        understandings: dict[str, AssetUnderstanding] = {}
        usable_assets: list[EmbeddingAsset] = []
        for asset in assets:
            try:
                understandings[asset.asset_id] = _stored_understanding(asset)
            except (TypeError, ValueError):
                continue
            usable_assets.append(asset)

        (
            subject_clusters,
            subject_member_groups,
            subject_cluster_status,
        ) = await self._ensure_subject_clusters(
            workspace_id=workspace_id,
            assets=assets,
        )
        input_revision = _input_revision(
            assets,
            subject_clusters=subject_clusters,
            subject_member_groups=subject_member_groups,
            candidate_policy=self._candidate_policy,
        )
        hierarchy_state = (
            await self._relation_repository.get_hierarchy_build_state(
                workspace_id=workspace_id,
            )
            if (not force_understanding and self._relation_repository is not None)
            else None
        )
        persisted = (
            await self._relation_repository.load(
                workspace_id=workspace_id,
                input_revision=input_revision,
            )
            if (
                not force_understanding
                and self._relation_repository is not None
                and hierarchy_state is not None
                and hierarchy_state.get("subject_cluster_status", {}).get("graph_policy")
                == "entity_hierarchy_v1"
            )
            else None
        )
        if persisted is not None and (persisted.get("asset_revisions") or not assets):
            return _hydrate_persisted_graph(
                workspace_id=workspace_id,
                assets=assets,
                usable_assets=usable_assets,
                understandings=understandings,
                understanding_errors=understanding_errors,
                persisted=persisted,
            )

        entity_candidates = await self._entity_candidates(
            workspace_id=workspace_id,
            assets=usable_assets,
            subject_clusters=subject_clusters,
            subject_member_groups=subject_member_groups,
        )
        candidate_vectors, embedding_model = await self._embed_entity_candidates(
            workspace_id=workspace_id,
            candidates=entity_candidates,
        )
        return await self._build_hierarchy_graph(
            workspace_id=workspace_id,
            assets=assets,
            usable_assets=usable_assets,
            understandings=understandings,
            understanding_errors=understanding_errors,
            subject_clusters=subject_clusters,
            subject_member_groups=subject_member_groups,
            subject_cluster_status=subject_cluster_status,
            entity_candidates=entity_candidates,
            candidate_vectors=candidate_vectors,
            embedding_model=embedding_model,
            force_understanding=force_understanding,
            input_revision=input_revision,
        )

    async def _build_hierarchy_graph(
        self,
        *,
        workspace_id: str,
        assets: Sequence[EmbeddingAsset],
        usable_assets: list[EmbeddingAsset],
        understandings: Mapping[str, AssetUnderstanding],
        understanding_errors: list[dict[str, str]],
        subject_clusters: Sequence[Any],
        subject_member_groups: Sequence[Sequence[Any]],
        subject_cluster_status: Mapping[str, Any],
        entity_candidates: Sequence[Mapping[str, Any]],
        candidate_vectors: Mapping[str, Sequence[float]],
        embedding_model: str,
        force_understanding: bool,
        input_revision: str,
    ) -> dict[str, Any]:
        """Synchronize governed candidates, then run the bounded hierarchy workflow."""

        repository = self._relation_repository
        if repository is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("relationship graph persistence is unavailable")
        hierarchy_status = {
            **subject_cluster_status,
            "graph_policy": "entity_hierarchy_v1",
            "eligible_cluster_modes": [
                ClusterMode.RESIDENT_OPEN.value,
                ClusterMode.RESIDENT_MANUAL.value,
            ],
        }
        state = await repository.get_hierarchy_build_state(workspace_id=workspace_id)
        existing_policy = (
            state.get("subject_cluster_status", {}).get("graph_policy")
            if state is not None
            else None
        )
        if state is None or existing_policy != "entity_hierarchy_v1":
            initialization = await repository.initialize_hierarchy_graph(
                workspace_id=workspace_id,
                input_revision=input_revision,
                subject_cluster_status=hierarchy_status,
                expected_build_version=(int(state["build_version"]) if state is not None else None),
                expected_input_revision=(
                    str(state["input_revision"]) if state is not None else None
                ),
            )
            if not initialization.initialized:
                # A concurrent build completed initialization after this call
                # captured its snapshot. Discard that stale snapshot rather
                # than synchronizing it over the newer authoritative set.
                refreshed_assets = await self._embedding_repository.list_assets(
                    workspace_id=workspace_id
                )
                return await self._build_once(
                    workspace_id=workspace_id,
                    force_understanding=force_understanding,
                    assets=refreshed_assets,
                )
            expected_build_version = initialization.build_version
            expected_input_revision = initialization.input_revision
        else:
            expected_build_version = int(state["build_version"])
            expected_input_revision = str(state["input_revision"])

        candidates = _hierarchy_entity_candidates(
            entity_candidates,
            candidate_vectors=candidate_vectors,
            embedding_model=embedding_model,
        )
        sync = await repository.sync_hierarchy_entity_candidates(
            workspace_id=workspace_id,
            sync=HierarchyCandidateSync(
                operation_id=_hierarchy_operation_id(
                    "candidate-sync",
                    workspace_id=workspace_id,
                    input_revision=input_revision,
                    build_version=expected_build_version,
                ),
                expected_build_version=expected_build_version,
                expected_input_revision=expected_input_revision,
                input_revision=input_revision,
                subject_cluster_status=hierarchy_status,
                candidates=candidates,
                governed_candidate_ids=[candidate.candidate_id for candidate in candidates],
                asset_revisions={asset.asset_id: _asset_revision(asset) for asset in usable_assets},
            ),
        )
        adapter = _HierarchyRepositoryAdapter(
            repository=repository,
            model_client=self._model_client,
            expected_build_version=sync.build_version,
            expected_input_revision=sync.input_revision,
            subject_cluster_status=hierarchy_status,
        )
        workflow_result = None
        if sync.pending_entity_ids:
            workflow = EntityHierarchyWorkflow(
                repository=adapter,
                agent=EntityHierarchyAgentAdapter(model=self._model_client),
                checkpointer=self._hierarchy_checkpointer,
            )
            workflow_result = await workflow.run(
                workspace_id=workspace_id,
                input_revision=sync.input_revision,
                job_id=_hierarchy_operation_id(
                    "hierarchy-job",
                    workspace_id=workspace_id,
                    input_revision=sync.input_revision,
                    build_version=sync.build_version,
                ),
                pending_entity_ids=sync.pending_entity_ids,
            )

        assignment_stats = await self._assign_unclustered_assets(
            workspace_id=workspace_id,
            assets=usable_assets,
            understandings=understandings,
            input_revision=sync.input_revision,
            hierarchy_status=hierarchy_status,
            expected_build_version=adapter.expected_build_version,
            expected_input_revision=adapter.expected_input_revision,
        )
        persisted = await repository.load_current(workspace_id=workspace_id)
        if persisted is None:  # pragma: no cover - persistence contract
            raise RuntimeError("hierarchy workflow completed without a persisted graph")
        graph = _hydrate_persisted_graph(
            workspace_id=workspace_id,
            assets=assets,
            usable_assets=usable_assets,
            understandings=dict(understandings),
            understanding_errors=understanding_errors,
            persisted=persisted,
        )
        graph.update(
            {
                "workspace_id": workspace_id,
                "source_asset_count": len(assets),
                "understood_asset_count": len(usable_assets),
                "understanding_errors": understanding_errors,
                "entity_candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key
                        not in {
                            "embedding_vector",
                            "embedding_model",
                            "embedding_source_hash",
                        }
                    }
                    for candidate in entity_candidates
                ],
                "subject_cluster_status": hierarchy_status,
                "entity_hierarchy": {
                    "pending_entity_count": len(sync.pending_entity_ids),
                    "workflow": (
                        workflow_result.model_dump(mode="json")
                        if workflow_result is not None
                        else None
                    ),
                    "asset_assignment": assignment_stats,
                },
                "persistent_cache_hit": False,
                "build_version": persisted.get("build_version"),
            }
        )
        return graph

    async def _assign_unclustered_assets(
        self,
        *,
        workspace_id: str,
        assets: Sequence[EmbeddingAsset],
        understandings: Mapping[str, AssetUnderstanding],
        input_revision: str,
        hierarchy_status: Mapping[str, Any],
        expected_build_version: int,
        expected_input_revision: str,
    ) -> dict[str, int]:
        """Assign only Assets that are absent from every subject cluster.

        Dynamic-cluster members are intentionally excluded here.  They stay in
        the retrieval/automatic-clustering layer and never become graph
        memberships through the fallback Agent path.
        """

        repository = self._relation_repository
        if repository is None:
            return {"unclustered_asset_count": 0, "agent_assignment_count": 0}
        clustered_asset_ids = await self._all_subject_cluster_member_ids(workspace_id=workspace_id)
        candidates_assets = [asset for asset in assets if asset.asset_id not in clustered_asset_ids]
        if not candidates_assets:
            return {"unclustered_asset_count": 0, "agent_assignment_count": 0}
        persisted = await repository.load_current(workspace_id=workspace_id)
        if persisted is None:
            return {
                "unclustered_asset_count": len(candidates_assets),
                "agent_assignment_count": 0,
            }
        requests = await self._recall_asset_assignment_requests(
            workspace_id=workspace_id,
            assets=candidates_assets,
            understandings=understandings,
            entities=persisted["entities"],
        )
        resolution = await self._assign_assets_to_entities(requests)
        decisions = {decision.asset_id: decision for decision in resolution.assignments}
        assignments = [
            HierarchyAssetAssignment(
                asset_id=asset.asset_id,
                entity_id=(
                    decisions[asset.asset_id].entity_id if asset.asset_id in decisions else None
                ),
                relation="AGENT_ASSIGNED",
                description=(
                    decisions[asset.asset_id].description if asset.asset_id in decisions else ""
                ),
                reason=(
                    decisions[asset.asset_id].reason
                    if asset.asset_id in decisions
                    else "no Entity passed the semantic recall threshold"
                ),
                content_subject=_primary_subject(understandings[asset.asset_id]),
            )
            for asset in candidates_assets
        ]
        result = await repository.apply_asset_assignments(
            workspace_id=workspace_id,
            operation_id=_hierarchy_operation_id(
                "asset-assignment",
                workspace_id=workspace_id,
                input_revision=input_revision,
                build_version=expected_build_version,
            ),
            expected_build_version=expected_build_version,
            expected_input_revision=expected_input_revision,
            input_revision=input_revision,
            subject_cluster_status=hierarchy_status,
            assignments=assignments,
            asset_revisions={asset.asset_id: _asset_revision(asset) for asset in candidates_assets},
        )
        return {
            "unclustered_asset_count": len(candidates_assets),
            "agent_assignment_count": sum(
                assignment.entity_id is not None for assignment in assignments
            ),
            "build_version": result.build_version,
        }

    async def _all_subject_cluster_member_ids(self, *, workspace_id: str) -> set[str]:
        return await self._subject_cluster_member_ids(workspace_id=workspace_id)

    async def _subject_cluster_member_ids(
        self,
        *,
        workspace_id: str,
        modes: Sequence[ClusterMode] | None = None,
    ) -> set[str]:
        repository = self._current_cluster_repository
        if repository is None:
            return set()
        clusters = await repository.list_clusters(
            workspace_id=workspace_id,
            embedding_type=EmbeddingType.SUBJECT_CONTENT.value,
            modes=modes,
        )
        member_groups = await asyncio.gather(
            *(
                repository.list_members(
                    cluster_id=cluster.cluster_id,
                    workspace_id=workspace_id,
                )
                for cluster in clusters
            )
        )
        return {str(member.asset_id) for members in member_groups for member in members}

    async def _recall_asset_assignment_requests(
        self,
        *,
        workspace_id: str,
        assets: Sequence[EmbeddingAsset],
        understandings: Mapping[str, AssetUnderstanding],
        entities: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        repository = self._current_cluster_repository
        vector_store = self._vector_store
        if not assets or repository is None or vector_store is None:
            return []
        list_embeddings = getattr(repository, "list_indexed_asset_embeddings", None)
        fetch_vectors = getattr(vector_store, "fetch_vectors", None)
        if not callable(list_embeddings) or not callable(fetch_vectors):
            return []
        indexed = await list_embeddings(
            workspace_id=workspace_id,
            embedding_type=EmbeddingType.SUBJECT_CONTENT.value,
            asset_ids=[asset.asset_id for asset in assets],
        )
        embedding_id_by_asset = {str(item.asset_id): str(item.embedding_id) for item in indexed}
        vectors = await fetch_vectors(list(embedding_id_by_asset.values()))
        normalized_entities = [
            (entity, vector)
            for entity in entities
            if (vector := _normalize_vector(entity.get("embedding_vector", []))) is not None
        ]
        asset_rows = _asset_display_graph(list(assets), dict(understandings))["assets"]
        rows_by_id = {row["asset_id"]: row for row in asset_rows}
        requests: list[dict[str, Any]] = []
        for asset in assets:
            embedding_id = embedding_id_by_asset.get(asset.asset_id)
            asset_vector = _normalize_vector(vectors.get(embedding_id, []))
            if asset_vector is None:
                continue
            recalled_entities = [
                entity
                for similarity, entity in sorted(
                    (
                        (_cosine_similarity(asset_vector, entity_vector), entity)
                        for entity, entity_vector in normalized_entities
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )[: self._incremental_entity_recall_top_k]
                if similarity >= self._incremental_entity_recall_similarity_threshold
            ]
            if not recalled_entities:
                continue
            row = rows_by_id[asset.asset_id]
            path_parts = Path(asset.source_relative_path).parts
            requests.append(
                {
                    "asset": {
                        "asset_id": asset.asset_id,
                        "metadata": {
                            "source_path": asset.source_relative_path,
                            "asset_name": row["asset_name"],
                            "entity_hints": [
                                {"value": part, "scope": "collection"} for part in path_parts[:-1]
                            ],
                        },
                        "content_description": row["asset_description"],
                        "content_subject": row["primary_subject"],
                    },
                    "candidates": [
                        {
                            "entity_id": str(entity["entity_id"]),
                            "name": str(entity["name"]),
                            "semantic": str(entity["semantic"]),
                        }
                        for entity in recalled_entities
                    ],
                }
            )
        return requests

    async def _assign_assets_to_entities(
        self,
        requests: Sequence[Mapping[str, Any]],
    ) -> AssetEntityAssignmentResolution:
        assign = getattr(self._model_client, "assign_assets_to_entities", None)
        if not requests or not callable(assign):
            return AssetEntityAssignmentResolution()
        resolutions: list[AssetEntityAssignmentResolution] = []
        for batch in _mapping_batches(requests, size=10):
            request = AssetEntityAssignmentRequest.model_validate({"assignments": batch})
            resolution = await assign(batch)
            request.validate_resolution(resolution)
            resolutions.append(resolution)
        return AssetEntityAssignmentResolution(
            assignments=[
                assignment for resolution in resolutions for assignment in resolution.assignments
            ]
        )

    async def update_assets(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
        affected_cluster_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Refresh the hierarchy graph after cluster membership changes."""

        repository = self._relation_repository
        requested_ids = set(asset_ids)
        if repository is None or not requested_ids:
            return {"status": "skipped", "updated_asset_count": 0}
        dynamic_member_ids = await self._subject_cluster_member_ids(
            workspace_id=workspace_id,
            modes=(ClusterMode.DYNAMIC,),
        )
        if requested_ids.issubset(dynamic_member_ids):
            return {
                "status": "skipped_dynamic_cluster_members",
                "updated_asset_count": 0,
            }
        graph = await self.build(workspace_id=workspace_id)
        return {
            "status": "updated",
            "updated_asset_count": len(requested_ids),
            "build_version": graph.get("build_version"),
            "entity_hierarchy": graph.get("entity_hierarchy"),
        }

    async def _load_subject_clusters(
        self,
        *,
        workspace_id: str,
    ) -> tuple[list[Any], list[list[Any]], dict[str, Any]]:
        repository = self._current_cluster_repository
        if repository is None:
            return [], [], {"status": "unavailable", "cluster_count": 0}
        clusters = list(
            await repository.list_clusters(
                workspace_id=workspace_id,
                embedding_type=EmbeddingType.SUBJECT_CONTENT.value,
                modes=(
                    ClusterMode.RESIDENT_OPEN,
                    ClusterMode.RESIDENT_MANUAL,
                ),
            )
        )
        member_groups = await asyncio.gather(
            *(
                repository.list_members(
                    cluster_id=cluster.cluster_id,
                    workspace_id=workspace_id,
                )
                for cluster in clusters
            )
        )
        return (
            clusters,
            list(member_groups),
            {
                "status": "available" if clusters else "empty",
                "cluster_count": len(clusters),
                "generated": False,
            },
        )

    async def _ensure_subject_clusters(
        self,
        *,
        workspace_id: str,
        assets: Sequence[EmbeddingAsset],
    ) -> tuple[list[Any], list[list[Any]], dict[str, Any]]:
        repository = self._current_cluster_repository
        if repository is None:
            return (
                [],
                [],
                {
                    "status": "unavailable",
                    "cluster_count": 0,
                    "message": "主体聚类仓库不可用",
                },
            )

        async def load() -> tuple[list[Any], list[list[Any]]]:
            clusters, members, _ = await self._load_subject_clusters(workspace_id=workspace_id)
            return clusters, members

        clusters, member_groups = await load()
        if clusters:
            return (
                clusters,
                member_groups,
                {
                    "status": "available",
                    "cluster_count": len(clusters),
                    "generated": False,
                },
            )

        # Dynamic clusters are deliberately excluded from the relationship
        # graph.  Do not create one as a side effect of an empty governed set.
        runner = None
        if runner is None:
            return (
                [],
                [],
                {
                    "status": "unavailable",
                    "cluster_count": 0,
                    "generated": False,
                    "message": "没有主体聚类结果，且聚类服务不可用",
                },
            )

        result = await runner.run(
            workspace_id=workspace_id,
            embedding_type=EmbeddingType.SUBJECT_CONTENT,
            trigger="relation_graph",
        )
        clusters, member_groups = await load()
        result_status = getattr(result, "status", "unknown")
        status_value = getattr(result_status, "value", str(result_status))
        error = getattr(result, "error", None)
        return (
            clusters,
            member_groups,
            {
                "status": "generated" if clusters else status_value,
                "cluster_count": len(clusters),
                "generated": True,
                "cluster_run_id": getattr(result, "cluster_run_id", None),
                "message": error,
            },
        )

    async def _entity_candidates(
        self,
        *,
        workspace_id: str,
        assets: list[EmbeddingAsset],
        subject_clusters: Sequence[Any],
        subject_member_groups: Sequence[Sequence[Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        usable_ids = {asset.asset_id for asset in assets}
        for cluster, members in zip(subject_clusters, subject_member_groups, strict=True):
            asset_ids = list(
                dict.fromkeys(
                    member.asset_id for member in members if member.asset_id in usable_ids
                )
            )
            candidates.append(
                {
                    "candidate_id": f"subject_cluster:{cluster.cluster_id}",
                    "origin": "subject_cluster",
                    "name": cluster.name,
                    "semantic": cluster.description,
                    "asset_ids": asset_ids,
                    "cluster_id": cluster.cluster_id,
                    "embedding_vector": list(getattr(cluster, "embedding_vector", ()) or ()),
                    "embedding_model": str(getattr(cluster, "embedding_model", "") or ""),
                    "embedding_source_hash": str(
                        getattr(cluster, "embedding_source_hash", "") or ""
                    ),
                }
            )
        return candidates

    async def _embed_entity_candidates(
        self,
        *,
        workspace_id: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, list[float]], str]:
        embed_text = getattr(self._model_client, "embed_text", None)
        if not candidates:
            return {}, ""

        vectors: dict[str, list[float]] = {}
        model = ""
        missing: list[Mapping[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            stored = _normalize_vector(candidate.get("embedding_vector", ()))
            if stored is None:
                missing.append(candidate)
                continue
            vectors[candidate_id] = stored
            model = model or str(candidate.get("embedding_model", ""))

        if not missing or not callable(embed_text):
            return vectors, model

        async def embed(candidate: Mapping[str, Any]) -> tuple[str, EmbeddingResult]:
            text = "\n".join(
                part
                for part in (
                    str(candidate.get("name", "")).strip(),
                    str(candidate.get("semantic", "")).strip(),
                )
                if part
            )
            return str(candidate["candidate_id"]), await embed_text(text)

        outcomes = await asyncio.gather(
            *(embed(candidate) for candidate in missing),
            return_exceptions=True,
        )
        backfills: list[tuple[str, list[float], str, str]] = []
        candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in missing}
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                continue
            candidate_id, result = outcome
            vector = _normalize_vector(result.vector)
            if vector is None:
                continue
            vectors[candidate_id] = vector
            model = model or result.model
            candidate = candidates_by_id[candidate_id]
            cluster_id = str(candidate.get("cluster_id", ""))
            if cluster_id:
                backfills.append(
                    (
                        cluster_id,
                        vector,
                        result.model,
                        _cluster_embedding_source_hash(
                            str(candidate.get("name", "")),
                            str(candidate.get("semantic", "")),
                            result.model,
                        ),
                    )
                )

        set_embedding = getattr(self._current_cluster_repository, "set_embedding", None)
        if callable(set_embedding) and backfills:
            await asyncio.gather(
                *(
                    set_embedding(
                        cluster_id=cluster_id,
                        workspace_id=workspace_id,
                        vector=vector,
                        model=embedding_model,
                        source_hash=source_hash,
                    )
                    for cluster_id, vector, embedding_model, source_hash in backfills
                )
            )
        return vectors, model

    @property
    def _candidate_policy(self) -> str:
        return (
            "resident_subject_clusters_entity_hierarchy_v1:"
            "asset_assignment_recall_v1:"
            f"{self._incremental_entity_recall_similarity_threshold:.4f}:"
            f"top{self._incremental_entity_recall_top_k}"
        )


def _supports_hierarchy_workflow(repository: object | None) -> bool:
    required = (
        "get_hierarchy_build_state",
        "initialize_hierarchy_graph",
        "sync_hierarchy_entity_candidates",
        "load_entity_nodes",
        "search_similar_entity_nodes",
        "load_complete_entity_trees",
        "load_workspace_tree",
        "commit_hierarchy_batch",
        "apply_asset_assignments",
    )
    return repository is not None and all(
        callable(getattr(repository, method, None)) for method in required
    )


def _hierarchy_operation_id(
    kind: str,
    *,
    workspace_id: str,
    input_revision: str,
    build_version: int,
) -> str:
    value = f"{kind}\0{workspace_id}\0{input_revision}\0{build_version}".encode()
    return f"{kind}:{hashlib.sha256(value).hexdigest()[:24]}"


def _hierarchy_entity_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    candidate_vectors: Mapping[str, Sequence[float]],
    embedding_model: str,
) -> list[HierarchyEntityCandidate]:
    prepared: list[HierarchyEntityCandidate] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        vector = _normalize_vector(candidate_vectors.get(candidate_id, ()))
        if vector is None:
            raise ValueError(f"governed Entity candidate {candidate_id} has no semantic embedding")
        prepared.append(
            HierarchyEntityCandidate(
                candidate_id=candidate_id,
                origin=str(candidate["origin"]),
                name=str(candidate["name"]),
                semantic=str(candidate["semantic"]),
                asset_ids=[str(asset_id) for asset_id in candidate["asset_ids"]],
                embedding_vector=vector,
                embedding_model=(str(candidate.get("embedding_model") or embedding_model)),
            )
        )
    return prepared


def _mapping_batches(
    values: Sequence[Mapping[str, Any]],
    *,
    size: int,
) -> list[list[Mapping[str, Any]]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _primary_subject(understanding: AssetUnderstanding) -> str:
    primary = max(
        understanding.features.subject_content.items,
        key=lambda item: item.salience,
        default=None,
    )
    return primary.subject if primary is not None else ""


def _direct_entity_resolution(
    candidates: Sequence[Mapping[str, Any]],
) -> MergedEntityResolution:
    """Use each clustering candidate as an Entity without model-side merging."""

    return MergedEntityResolution.model_validate(
        {
            "entities": [
                {
                    "name": str(candidate.get("name", "")),
                    "semantic": str(candidate.get("semantic", "")),
                    "candidate_ids": [str(candidate["candidate_id"])],
                    "build_entity": True,
                    "reason": "主体聚类候选直接形成 Entity。",
                }
                for candidate in candidates
            ]
        }
    )


def _normalize_vector(vector: Sequence[float]) -> list[float] | None:
    if not vector or any(not math.isfinite(value) for value in vector):
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return None
    return [float(value / norm) for value in vector]


def _cluster_embedding_source_hash(name: str, description: str, model: str) -> str:
    payload = "\0".join((name.strip(), description.strip(), model.strip())).encode()
    return hashlib.sha256(payload).hexdigest()


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def _stored_understanding(asset: EmbeddingAsset) -> AssetUnderstanding:
    return AssetUnderstanding.model_validate(
        {
            "asset_name": Path(asset.file_name).stem or "未命名素材",
            "asset_description": asset.asset_description or "",
            "features": asset.asset_features,
        }
    )


def _asset_snapshot(assets: Sequence[EmbeddingAsset]) -> str:
    payload = [
        {
            "asset_id": asset.asset_id,
            "content_hash": asset.content_hash,
            "embedding_revision": asset.embedding_revision,
            "source_relative_path": asset.source_relative_path,
            "asset_description": asset.asset_description,
            "asset_features": asset.asset_features,
        }
        for asset in assets
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _asset_revision(asset: EmbeddingAsset) -> str:
    payload = {
        "content_hash": asset.content_hash,
        "embedding_revision": asset.embedding_revision,
        "source_relative_path": asset.source_relative_path,
        "asset_description": asset.asset_description,
        "asset_features": asset.asset_features,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _input_revision(
    assets: Sequence[EmbeddingAsset],
    *,
    subject_clusters: Sequence[Any],
    subject_member_groups: Sequence[Sequence[Any]],
    candidate_policy: str,
) -> str:
    cluster_payload = [
        {
            "cluster_id": cluster.cluster_id,
            "name": cluster.name,
            "description": cluster.description,
            "source_run_id": getattr(cluster, "source_run_id", None),
            "updated_at": getattr(cluster, "updated_at", None),
            "members": sorted(member.asset_id for member in members),
        }
        for cluster, members in zip(subject_clusters, subject_member_groups, strict=True)
    ]
    payload = {
        "asset_revision": _asset_snapshot(assets),
        "subject_clusters": cluster_payload,
        "relation_candidate_policy": candidate_policy,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _refresh_asset_entity_memberships(graph: dict[str, Any]) -> None:
    """Derive Asset display memberships from the final persisted relations."""

    assets_by_id = {item["asset_id"]: item for item in graph["assets"]}
    entities_by_id = {item["entity_id"]: item for item in graph["entities"]}
    for asset in assets_by_id.values():
        asset["main_entities"] = []
    for edge in graph["edges"]:
        asset = assets_by_id.get(edge["source"])
        entity = entities_by_id.get(edge["target"])
        if asset is None or entity is None:
            continue
        asset["main_entities"].append(
            {
                "subject": entity["name"],
                "description": entity["semantic"],
                "salience": 1.0,
                "origins": entity.get("origins", []),
            }
        )


def _asset_display_graph(
    assets: Sequence[EmbeddingAsset],
    understandings: Mapping[str, AssetUnderstanding],
) -> dict[str, Any]:
    """Build display-only asset rows; entity relations come only from storage."""
    rows: list[dict[str, Any]] = []
    for asset in assets:
        understanding = understandings[asset.asset_id]
        primary = max(
            understanding.features.subject_content.items,
            key=lambda item: item.salience,
            default=None,
        )
        rows.append(
            {
                "asset_id": asset.asset_id,
                "source_path": asset.source_relative_path,
                "asset_name": understanding.asset_name,
                "asset_description": understanding.asset_description,
                "primary_subject": (
                    {"subject": primary.subject, "description": primary.description}
                    if primary is not None
                    else None
                ),
                "main_entities": [],
                "subjects": [
                    {
                        "subject": item.subject,
                        "description": item.description,
                        "salience": item.salience,
                        "status": item.status.value,
                    }
                    for item in understanding.features.subject_content.items
                ],
            }
        )
    return {"assets": rows, "entities": [], "edges": [], "entity_edges": []}


def _hydrate_persisted_graph(
    *,
    workspace_id: str,
    assets: Sequence[EmbeddingAsset],
    usable_assets: list[EmbeddingAsset],
    understandings: dict[str, AssetUnderstanding],
    understanding_errors: list[dict[str, str]],
    persisted: Mapping[str, Any],
) -> dict[str, Any]:
    graph = _asset_display_graph(usable_assets, understandings)
    usable_ids = {asset.asset_id for asset in usable_assets}
    graph["entities"] = list(persisted["entities"])
    graph["edges"] = [edge for edge in persisted["edges"] if edge["source"] in usable_ids]
    graph["entity_edges"] = list(persisted.get("entity_edges", []))
    graph["rejected_relations"] = [
        relation
        for relation in persisted["rejected_relations"]
        if relation["source_id"] in usable_ids
    ]
    member_ids_by_entity: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        member_ids_by_entity[edge["target"]].append(edge["source"])
    retained_entity_ids = {
        entity_id for entity_id, member_ids in member_ids_by_entity.items() if member_ids
    }
    retained_entity_ids.update(
        str(edge[key])
        for edge in graph["entity_edges"]
        for key in ("source_entity_id", "target_entity_id")
    )
    graph["entities"] = [
        entity for entity in graph["entities"] if entity["entity_id"] in retained_entity_ids
    ]
    graph["edges"] = [edge for edge in graph["edges"] if edge["target"] in retained_entity_ids]
    graph["entity_edges"] = [
        edge
        for edge in graph["entity_edges"]
        if edge["source_entity_id"] in retained_entity_ids
        and edge["target_entity_id"] in retained_entity_ids
    ]
    for entity in graph["entities"]:
        entity["asset_ids"] = member_ids_by_entity[entity["entity_id"]]
    _refresh_asset_entity_memberships(graph)
    graph.update(
        {
            "workspace_id": workspace_id,
            "source_asset_count": len(assets),
            "understood_asset_count": len(usable_assets),
            "understanding_errors": understanding_errors,
            "metadata_content_relations": {},
            "entity_candidates": [],
            "merged_entity_decisions": {"entities": []},
            "subject_cluster_status": persisted["subject_cluster_status"],
            "build_version": persisted["build_version"],
            "persistent_cache_hit": True,
            "incremental_update_pending_asset_ids": [],
        }
    )
    graph["entity_count"] = len(graph["entities"])
    graph["entity_edge_count"] = len(graph["entity_edges"])
    graph["edge_count"] = len(graph["edges"]) + len(graph["entity_edges"])
    return graph


def _empty_graph(workspace_id: str) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "source_asset_count": 0,
        "understood_asset_count": 0,
        "understanding_errors": [],
        "metadata_content_relations": {},
        "asset_count": 0,
        "entity_count": 0,
        "entity_edge_count": 0,
        "edge_count": 0,
        "assets": [],
        "entities": [],
        "edges": [],
        "entity_edges": [],
    }
