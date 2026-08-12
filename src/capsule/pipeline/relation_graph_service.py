"""Build one workspace relationship graph from persisted Asset understanding."""

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from capsule.db.repositories import EmbeddingAsset, EmbeddingRepository
from capsule.enums import ClusterMode, EmbeddingType
from capsule.pipeline.understanding import AssetUnderstandingService
from capsule.relation_graph import (
    AssetEntityRelationResolution,
    MergedEntityResolution,
    MetadataContentResolution,
    _entity_key,
    _metadata_names,
    apply_asset_entity_relations,
    build_merged_candidate_graph,
    build_relation_graph,
)
from capsule.schemas import AssetUnderstanding, EmbeddingResult
from capsule.search.models import SearchFilters, VectorSearchHit


class RelationResolutionClient(Protocol):
    async def resolve_metadata_content_entities(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        guidance: str | None = None,
    ) -> MetadataContentResolution: ...

    async def generate_asset_entity_relations(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> AssetEntityRelationResolution: ...

    async def merge_entity_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> MergedEntityResolution: ...

    async def embed_text(self, text: str) -> EmbeddingResult: ...


class AssetVectorSearch(Protocol):
    async def search_raw(
        self,
        *,
        vector: list[float],
        workspace_id: str,
        embedding_type: str,
        filters: SearchFilters,
        limit: int,
    ) -> list[VectorSearchHit]: ...


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


class SubjectClusterRunner(Protocol):
    async def run(
        self,
        *,
        workspace_id: str,
        embedding_type: EmbeddingType,
        trigger: str = "user",
    ) -> Any: ...


class RelationPersistenceRepository(Protocol):
    async def load(
        self,
        *,
        workspace_id: str,
        input_revision: str,
    ) -> dict[str, Any] | None: ...

    async def load_current(self, *, workspace_id: str) -> dict[str, Any] | None: ...

    async def replace(
        self,
        *,
        workspace_id: str,
        input_revision: str,
        graph: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        subject_cluster_status: Mapping[str, Any],
        asset_revisions: Mapping[str, str],
    ) -> int: ...

    async def update_assets(
        self,
        *,
        workspace_id: str,
        input_revision: str,
        asset_ids: Sequence[str],
        resolution: AssetEntityRelationResolution,
        subject_cluster_status: Mapping[str, Any],
        affected_source_members: Mapping[str, Sequence[str]],
        asset_revisions: Mapping[str, str],
    ) -> int | None: ...


class RelationGraphService:
    def __init__(
        self,
        *,
        embedding_repository: EmbeddingRepository,
        understanding_service: AssetUnderstandingService,
        model_client: RelationResolutionClient,
        current_cluster_repository: SubjectClusterRepository | None = None,
        subject_cluster_runner: SubjectClusterRunner | None = None,
        relation_repository: RelationPersistenceRepository | None = None,
        vector_store: AssetVectorSearch | None = None,
        metadata_candidates_enabled: bool = False,
        merge_candidates_enabled: bool = True,
        entity_merge_similarity_threshold: float = 0.67,
        asset_recall_similarity_threshold: float = 0.55,
        asset_recall_top_k: int = 20,
        asset_recall_path_boost: float = 0.25,
    ) -> None:
        self._embedding_repository = embedding_repository
        self._understanding_service = understanding_service
        self._model_client = model_client
        self._current_cluster_repository = current_cluster_repository
        self._subject_cluster_runner = subject_cluster_runner
        self._relation_repository = relation_repository
        self._vector_store = vector_store
        self._metadata_candidates_enabled = metadata_candidates_enabled
        self._merge_candidates_enabled = merge_candidates_enabled
        self._entity_merge_similarity_threshold = entity_merge_similarity_threshold
        self._asset_recall_similarity_threshold = asset_recall_similarity_threshold
        self._asset_recall_top_k = asset_recall_top_k
        self._asset_recall_path_boost = asset_recall_path_boost
        self._build_tasks: dict[tuple[str, bool, str], asyncio.Task[dict[str, Any]]] = {}
        self._build_tasks_lock = asyncio.Lock()

    async def build(
        self,
        *,
        workspace_id: str,
        force_understanding: bool = False,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        assets = await self._embedding_repository.list_assets(workspace_id=workspace_id)
        snapshot = _asset_snapshot(assets)
        key = (workspace_id, force_understanding or force_rebuild, snapshot)
        async with self._build_tasks_lock:
            task = self._build_tasks.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._build_once(
                        workspace_id=workspace_id,
                        force_understanding=force_understanding,
                        force_rebuild=force_rebuild,
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
        force_rebuild: bool,
        assets: list[EmbeddingAsset],
    ) -> dict[str, Any]:
        if not assets:
            return _empty_graph(workspace_id)

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
        persisted = (
            await self._relation_repository.load(
                workspace_id=workspace_id,
                input_revision=input_revision,
            )
            if (
                not force_understanding
                and not force_rebuild
                and self._relation_repository is not None
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

        if not force_understanding and not force_rebuild and self._relation_repository:
            current = await self._relation_repository.load_current(workspace_id=workspace_id)
            stored_revisions = current.get("asset_revisions", {}) if current else {}
            current_revisions = {asset.asset_id: _asset_revision(asset) for asset in assets}
            removed_ids = set(stored_revisions) - set(current_revisions)
            pending_ids = [
                asset_id
                for asset_id, revision in current_revisions.items()
                if stored_revisions.get(asset_id) != revision
            ]
            if current is not None and stored_revisions and not removed_ids and pending_ids:
                graph = _hydrate_persisted_graph(
                    workspace_id=workspace_id,
                    assets=assets,
                    usable_assets=usable_assets,
                    understandings=understandings,
                    understanding_errors=understanding_errors,
                    persisted=current,
                )
                graph["incremental_update_pending_asset_ids"] = pending_ids
                return graph

        decisions = (
            await self._resolve_relations(usable_assets, understandings)
            if self._metadata_candidates_enabled
            else {}
        )
        entity_candidates = await self._entity_candidates(
            workspace_id=workspace_id,
            assets=usable_assets,
            metadata_decisions=decisions,
            subject_clusters=subject_clusters,
            subject_member_groups=subject_member_groups,
        )
        candidate_vectors, embedding_model = await self._embed_entity_candidates(
            workspace_id=workspace_id,
            candidates=entity_candidates,
        )
        merged_entities = await self._merge_similar_entity_candidates(
            entity_candidates,
            candidate_vectors,
        )
        graph = build_merged_candidate_graph(
            usable_assets,
            understandings,
            candidates=entity_candidates,
            resolution=merged_entities,
        )
        _attach_entity_embeddings(
            graph,
            candidate_vectors=candidate_vectors,
            embedding_model=embedding_model,
        )
        recalled_edge_count = await self._recall_asset_entity_candidates(
            workspace_id=workspace_id,
            graph=graph,
        )
        edge_candidates = _asset_entity_candidates(graph)
        if edge_candidates:
            edge_resolution = await self._generate_edge_relations(edge_candidates)
            apply_asset_entity_relations(graph, edge_resolution)
        _refresh_asset_entity_memberships(graph)
        graph.update(
            {
                "workspace_id": workspace_id,
                "source_asset_count": len(assets),
                "understood_asset_count": len(usable_assets),
                "understanding_errors": understanding_errors,
                "metadata_content_relations": decisions,
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
                "merged_entity_decisions": merged_entities.model_dump(mode="json"),
                "entity_merge_similarity_threshold": (self._entity_merge_similarity_threshold),
                "asset_recall_similarity_threshold": (self._asset_recall_similarity_threshold),
                "recalled_edge_candidate_count": recalled_edge_count,
                "subject_cluster_status": subject_cluster_status,
                "persistent_cache_hit": False,
            }
        )
        if self._relation_repository is not None:
            graph["build_version"] = await self._relation_repository.replace(
                workspace_id=workspace_id,
                input_revision=input_revision,
                graph=graph,
                candidates=entity_candidates,
                subject_cluster_status=subject_cluster_status,
                asset_revisions={asset.asset_id: _asset_revision(asset) for asset in assets},
            )
        return graph

    async def update_assets(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str],
        affected_cluster_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Judge only new/changed Assets against existing affected Entities."""

        repository = self._relation_repository
        requested_ids = set(asset_ids)
        if repository is None or not requested_ids:
            return {"status": "skipped", "updated_asset_count": 0}
        persisted = await repository.load_current(workspace_id=workspace_id)
        if persisted is None:
            return {"status": "deferred", "updated_asset_count": 0}

        assets = await self._embedding_repository.list_assets(workspace_id=workspace_id)
        selected_assets = [asset for asset in assets if asset.asset_id in requested_ids]
        understandings: dict[str, AssetUnderstanding] = {}
        usable_assets: list[EmbeddingAsset] = []
        for asset in selected_assets:
            try:
                understandings[asset.asset_id] = _stored_understanding(asset)
            except (TypeError, ValueError):
                continue
            usable_assets.append(asset)
        if not usable_assets:
            return {"status": "deferred", "updated_asset_count": 0}

        subject_clusters, member_groups, cluster_status = await self._load_subject_clusters(
            workspace_id=workspace_id
        )
        affected_candidate_ids = {
            f"subject_cluster:{cluster_id}" for cluster_id in affected_cluster_ids
        }
        if self._metadata_candidates_enabled:
            for asset in usable_assets:
                affected_candidate_ids.update(
                    f"metadata:{_entity_key(name)}" for name in _metadata_names(asset)
                )
        previously_related = {
            item["target"] for item in persisted["edges"] if item["source"] in requested_ids
        } | {
            item["target_id"]
            for item in persisted["rejected_relations"]
            if item["source_id"] in requested_ids
        }
        affected_entities = [
            entity
            for entity in persisted["entities"]
            if entity["entity_id"] in previously_related
            or affected_candidate_ids.intersection(entity.get("candidate_ids", []))
        ]
        candidates = _direct_asset_entity_candidates(
            usable_assets,
            understandings,
            affected_entities,
        )
        resolution = (
            await self._generate_edge_relations(candidates)
            if candidates
            else AssetEntityRelationResolution()
        )
        input_revision = _input_revision(
            assets,
            subject_clusters=subject_clusters,
            subject_member_groups=member_groups,
            candidate_policy=self._candidate_policy,
        )
        members_by_source = {
            f"subject_cluster:{cluster.cluster_id}": [member.asset_id for member in members]
            for cluster, members in zip(subject_clusters, member_groups, strict=True)
            if cluster.cluster_id in set(affected_cluster_ids)
        }
        build_version = await repository.update_assets(
            workspace_id=workspace_id,
            input_revision=input_revision,
            asset_ids=[asset.asset_id for asset in usable_assets],
            resolution=resolution,
            subject_cluster_status=cluster_status,
            affected_source_members=members_by_source,
            asset_revisions={asset.asset_id: _asset_revision(asset) for asset in usable_assets},
        )
        return {
            "status": "updated" if build_version is not None else "deferred",
            "updated_asset_count": len(usable_assets),
            "candidate_entity_count": len(affected_entities),
            "build_version": build_version,
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

        # If the same asset revision has already completed a metadata-only
        # relationship build, do not run an unsuccessful clustering attempt on
        # every graph read. A changed asset snapshot produces a new revision and
        # permits a new attempt.
        if self._relation_repository is not None:
            empty_revision = _input_revision(
                assets,
                subject_clusters=[],
                subject_member_groups=[],
                candidate_policy=self._candidate_policy,
            )
            previous = await self._relation_repository.load(
                workspace_id=workspace_id,
                input_revision=empty_revision,
            )
            if previous is not None:
                previous_status = dict(previous["subject_cluster_status"])
                previous_status["generated"] = False
                previous_status["reused_attempt"] = True
                return [], [], previous_status

        runner = self._subject_cluster_runner
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
        metadata_decisions: Mapping[str, object],
        subject_clusters: Sequence[Any],
        subject_member_groups: Sequence[Sequence[Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        if self._metadata_candidates_enabled:
            metadata_members: dict[str, list[str]] = defaultdict(list)
            metadata_names: dict[str, str] = {}
            for asset in assets:
                for name in _metadata_names(asset):
                    key = _entity_key(name)
                    metadata_names.setdefault(key, name)
                    metadata_members[key].append(asset.asset_id)
            for key, asset_ids in metadata_members.items():
                unique_asset_ids = list(dict.fromkeys(asset_ids))
                if len(unique_asset_ids) < 2:
                    continue
                decision = metadata_decisions.get(key)
                decision_data = decision if isinstance(decision, Mapping) else {}
                candidates.append(
                    {
                        "candidate_id": f"metadata:{key}",
                        "origin": "metadata",
                        "name": metadata_names[key],
                        "semantic": str(
                            decision_data.get("entity_semantic") or metadata_names[key]
                        ),
                        "asset_ids": unique_asset_ids,
                        "evidence": decision_data,
                    }
                )

        usable_ids = {asset.asset_id for asset in assets}
        for cluster, members in zip(subject_clusters, subject_member_groups, strict=True):
            asset_ids = list(
                dict.fromkeys(
                    member.asset_id for member in members if member.asset_id in usable_ids
                )
            )
            if len(asset_ids) < 2:
                continue
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

    async def _generate_edge_relations(
        self,
        candidates: list[dict[str, Any]],
    ) -> AssetEntityRelationResolution:
        return await self._model_client.generate_asset_entity_relations(candidates)

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

    async def _merge_similar_entity_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        candidate_vectors: Mapping[str, Sequence[float]],
    ) -> MergedEntityResolution:
        if not candidates:
            return MergedEntityResolution()
        if not self._merge_candidates_enabled or not candidate_vectors:
            return _direct_entity_resolution(candidates)

        candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
        components = _similar_candidate_components(
            tuple(candidates_by_id),
            candidate_vectors,
            threshold=self._entity_merge_similarity_threshold,
        )
        merge_components = [component for component in components if len(component) > 1]
        merge_results = await asyncio.gather(
            *(
                self._model_client.merge_entity_candidates(
                    [candidates_by_id[candidate_id] for candidate_id in component]
                )
                for component in merge_components
            )
        )
        decisions = [decision for resolution in merge_results for decision in resolution.entities]
        resolved_ids = {
            candidate_id
            for decision in decisions
            for candidate_id in decision.candidate_ids
            if candidate_id in candidates_by_id
        }
        unresolved = [
            candidate
            for candidate_id, candidate in candidates_by_id.items()
            if candidate_id not in resolved_ids
        ]
        decisions.extend(_direct_entity_resolution(unresolved).entities)
        return MergedEntityResolution(entities=decisions)

    async def _recall_asset_entity_candidates(
        self,
        *,
        workspace_id: str,
        graph: dict[str, Any],
    ) -> int:
        if self._vector_store is None or not graph["entities"]:
            return 0

        searchable_entities = [
            entity for entity in graph["entities"] if entity.get("embedding_vector")
        ]
        outcomes = await asyncio.gather(
            *(
                self._vector_store.search_raw(
                    vector=list(entity["embedding_vector"]),
                    workspace_id=workspace_id,
                    embedding_type=EmbeddingType.SUBJECT_CONTENT.value,
                    filters=SearchFilters(),
                    limit=self._asset_recall_top_k,
                )
                for entity in searchable_entities
            ),
            return_exceptions=True,
        )
        asset_ids = {asset["asset_id"] for asset in graph["assets"]}
        assets_by_id = {asset["asset_id"]: asset for asset in graph["assets"]}
        existing = {(edge["source"], edge["target"]) for edge in graph["edges"]}
        recalled = 0
        for entity, outcome in zip(searchable_entities, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                continue
            for hit in outcome:
                edge_key = (hit.asset_id, entity["entity_id"])
                asset = assets_by_id.get(hit.asset_id)
                path_affinity = bool(
                    asset
                    and _entity_name_matches_path(str(entity["name"]), str(asset["source_path"]))
                )
                recall_confidence = min(
                    1.0,
                    hit.similarity + (self._asset_recall_path_boost if path_affinity else 0.0),
                )
                if (
                    hit.asset_id not in asset_ids
                    or recall_confidence < self._asset_recall_similarity_threshold
                    or edge_key in existing
                ):
                    continue
                graph["edges"].append(
                    {
                        "source": hit.asset_id,
                        "target": entity["entity_id"],
                        "relation": "VECTOR_RECALL_CANDIDATE",
                        "description": "",
                        "recall_similarity": hit.similarity,
                        "recall_confidence": recall_confidence,
                        "path_affinity": path_affinity,
                    }
                )
                existing.add(edge_key)
                recalled += 1
        graph["edge_count"] = len(graph["edges"])
        return recalled

    async def _resolve_relations(
        self,
        assets: list[EmbeddingAsset],
        understandings: dict[str, AssetUnderstanding],
    ) -> dict[str, object]:
        names_by_asset = {asset.asset_id: _metadata_names(asset) for asset in assets}
        counts = Counter(_entity_key(name) for names in names_by_asset.values() for name in names)
        display_names: dict[str, str] = {}
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for asset in assets:
            items = understandings[asset.asset_id].features.subject_content.items
            primary = max(items, key=lambda item: item.salience, default=None)
            if primary is None:
                continue
            for name in names_by_asset[asset.asset_id]:
                key = _entity_key(name)
                if counts[key] < 2:
                    continue
                display_names.setdefault(key, name)
                grouped[key].append(
                    {
                        "asset_id": asset.asset_id,
                        "source_path": asset.source_relative_path,
                        "content_subject": primary.subject,
                        "content_description": primary.description,
                    }
                )
        if not grouped:
            return {}
        resolution = await self._model_client.resolve_metadata_content_entities(
            [
                {
                    "metadata_entity": display_names[key],
                    "assets": members,
                }
                for key, members in grouped.items()
            ]
        )
        return {
            _entity_key(item.metadata_entity): item.model_dump(mode="json")
            for item in resolution.decisions
        }

    @property
    def _candidate_policy(self) -> str:
        source_policy = (
            "metadata_and_subject_clusters_v2"
            if self._metadata_candidates_enabled
            else "subject_clusters_only_v2"
        )
        merge_policy = "similarity_gated_merge_v1" if self._merge_candidates_enabled else "unmerged"
        return (
            f"{source_policy}:{merge_policy}:"
            f"{self._entity_merge_similarity_threshold:.4f}:"
            f"asset_recall_v1:{self._asset_recall_similarity_threshold:.4f}:"
            f"top{self._asset_recall_top_k}:pathboost{self._asset_recall_path_boost:.4f}"
        )


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


def _similar_candidate_components(
    candidate_ids: Sequence[str],
    vectors: Mapping[str, Sequence[float]],
    *,
    threshold: float,
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {candidate_id: set() for candidate_id in candidate_ids}
    for index, left_id in enumerate(candidate_ids):
        left = vectors.get(left_id)
        if left is None:
            continue
        for right_id in candidate_ids[index + 1 :]:
            right = vectors.get(right_id)
            if right is None or _cosine_similarity(left, right) < threshold:
                continue
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)

    components: list[list[str]] = []
    visited: set[str] = set()
    for candidate_id in candidate_ids:
        if candidate_id in visited:
            continue
        pending = [candidate_id]
        component: list[str] = []
        visited.add(candidate_id)
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                pending.append(neighbor)
        components.append(component)
    return components


def _attach_entity_embeddings(
    graph: dict[str, Any],
    *,
    candidate_vectors: Mapping[str, Sequence[float]],
    embedding_model: str,
) -> None:
    for entity in graph["entities"]:
        member_vectors = [
            candidate_vectors[candidate_id]
            for candidate_id in entity.get("candidate_ids", [])
            if candidate_id in candidate_vectors
        ]
        if not member_vectors:
            entity["embedding_vector"] = []
            entity["embedding_model"] = ""
            continue
        average = [
            sum(values) / len(member_vectors) for values in zip(*member_vectors, strict=True)
        ]
        entity["embedding_vector"] = _normalize_vector(average) or []
        entity["embedding_model"] = embedding_model


def _entity_name_matches_path(entity_name: str, source_path: str) -> bool:
    entity_key = _entity_key(entity_name)
    for part in Path(source_path).parts[:-1]:
        part_key = _entity_key(part)
        if len(part_key) >= 2 and (part_key in entity_key or entity_key in part_key):
            return True
    return False


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


def _asset_entity_candidates(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    assets_by_id = {item["asset_id"]: item for item in graph["assets"]}
    entities_by_id = {item["entity_id"]: item for item in graph["entities"]}
    candidates: list[dict[str, Any]] = []
    for edge in graph["edges"]:
        source = assets_by_id.get(edge["source"])
        target = entities_by_id.get(edge["target"])
        if source is None or target is None:
            continue
        path_parts = Path(source["source_path"]).parts
        candidates.append(
            {
                "source_id": edge["source"],
                "target_id": edge["target"],
                "asset": {
                    "metadata": {
                        "source_path": source["source_path"],
                        "asset_name": source["asset_name"],
                        "entity_hints": [
                            {"value": part, "scope": "collection"} for part in path_parts[:-1]
                        ],
                    },
                    "content_description": source["asset_description"],
                    "content_subject": source["primary_subject"],
                },
                "entity": {
                    "name": target["name"],
                    "semantic": target["semantic"],
                },
            }
        )
    return candidates


def _direct_asset_entity_candidates(
    assets: Sequence[EmbeddingAsset],
    understandings: Mapping[str, AssetUnderstanding],
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = build_relation_graph(list(assets), dict(understandings))["assets"]
    asset_rows = {row["asset_id"]: row for row in rows}
    candidates: list[dict[str, Any]] = []
    for asset in assets:
        row = asset_rows[asset.asset_id]
        path_parts = Path(asset.source_relative_path).parts
        for entity in entities:
            candidates.append(
                {
                    "source_id": asset.asset_id,
                    "target_id": entity["entity_id"],
                    "asset": {
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
                    "entity": {
                        "name": entity["name"],
                        "semantic": entity["semantic"],
                    },
                }
            )
    return candidates


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


def _hydrate_persisted_graph(
    *,
    workspace_id: str,
    assets: Sequence[EmbeddingAsset],
    usable_assets: list[EmbeddingAsset],
    understandings: dict[str, AssetUnderstanding],
    understanding_errors: list[dict[str, str]],
    persisted: Mapping[str, Any],
) -> dict[str, Any]:
    graph = build_relation_graph(usable_assets, understandings)
    usable_ids = {asset.asset_id for asset in usable_assets}
    graph["entities"] = list(persisted["entities"])
    graph["edges"] = [edge for edge in persisted["edges"] if edge["source"] in usable_ids]
    graph["rejected_relations"] = [
        relation
        for relation in persisted["rejected_relations"]
        if relation["source_id"] in usable_ids
    ]
    member_ids_by_entity: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        member_ids_by_entity[edge["target"]].append(edge["source"])
    graph["entities"] = [
        entity
        for entity in graph["entities"]
        if len(member_ids_by_entity[entity["entity_id"]]) >= 2
    ]
    retained_entity_ids = {entity["entity_id"] for entity in graph["entities"]}
    graph["edges"] = [edge for edge in graph["edges"] if edge["target"] in retained_entity_ids]
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
    graph["edge_count"] = len(graph["edges"])
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
        "edge_count": 0,
        "assets": [],
        "entities": [],
        "edges": [],
    }
