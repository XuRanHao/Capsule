"""Transactional persistence for source files, assets, jobs, and Embeddings."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from capsule.db.base import id_factory
from capsule.db.models import (
    Asset,
    EmbeddingRecord,
    GraphAsset,
    GraphAssetBinding,
    LogicalEntity,
    LogicalEntityRelation,
    ModelCallLog,
    NarrativeGraph,
    ProcessingJob,
    QueryImageUpload,
    SourceFile,
    Workspace,
)
from capsule.db.session import Database
from capsule.enums import (
    AssetIndexRole,
    AssetNameSource,
    AssetType,
    EmbeddingStatus,
    JobStatus,
    PipelineStage,
    ProcessingStatus,
)
from capsule.features import embedding_channel_is_eligible
from capsule.schemas import (
    AssetCreate,
    AssetEmbeddingState,
    AssetListResponse,
    AssetSourceRecord,
    AssetUnderstanding,
    AssetViewRecord,
    DiscoveredFile,
    ProcessingJobRecord,
    StoredFileResult,
    WorkspaceRecord,
)


@dataclass(slots=True, frozen=True)
class AssetMediaTarget:
    asset_id: str
    workspace_id: str
    asset_type: str
    source_storage_uri: str
    source_mime_type: str
    preview_uri: str | None
    derived_file_uri: str | None


class RelationGraphRepository:
    """Transactional persistence for the selected narrative graph."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def load_current_graph_context(
        self,
        *,
        workspace_id: str,
        graph_id: str,
    ) -> dict[str, Any]:
        """Load the small, user-selected graph context for an Agent turn."""

        async with self._database.session() as session:
            graph = await session.scalar(
                select(NarrativeGraph).where(
                    NarrativeGraph.graph_id == graph_id,
                    NarrativeGraph.workspace_id == workspace_id,
                )
            )
            if graph is None:
                raise ValueError("narrative graph does not exist in workspace")
            entities = list(
                await session.scalars(
                    select(LogicalEntity)
                    .where(LogicalEntity.graph_id == graph_id)
                    .order_by(LogicalEntity.entity_id)
                )
            )
            relations = list(
                await session.scalars(
                    select(LogicalEntityRelation)
                    .where(LogicalEntityRelation.graph_id == graph_id)
                    .order_by(LogicalEntityRelation.relation_id)
                )
            )
        return {
            "graph": {
                "graph_id": graph.graph_id,
                "workspace_id": graph.workspace_id,
                "name": graph.name,
                "description": graph.description,
                "narrative_context": dict(graph.narrative_context),
            },
            "entities": [_entity_node_payload(entity) for entity in entities],
            "relations": [_relation_payload(relation) for relation in relations],
        }

    async def get_entity_detail(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        entity_id: str,
    ) -> dict[str, Any]:
        """Load one logical entity and the Assets currently bound to it."""

        async with self._database.session() as session:
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            entity = await session.scalar(
                select(LogicalEntity).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == entity_id,
                )
            )
            if entity is None:
                raise ValueError("entity does not exist in graph")
            assets = list(
                await session.scalars(
                    select(Asset)
                    .join(
                        GraphAsset,
                        (GraphAsset.asset_id == Asset.asset_id)
                        & (GraphAsset.workspace_id == Asset.workspace_id),
                    )
                    .join(
                        GraphAssetBinding,
                        (GraphAssetBinding.graph_id == GraphAsset.graph_id)
                        & (GraphAssetBinding.asset_id == GraphAsset.asset_id),
                    )
                    .where(
                        GraphAsset.graph_id == graph_id,
                        GraphAsset.workspace_id == workspace_id,
                        GraphAssetBinding.entity_id == entity_id,
                    )
                    .order_by(Asset.asset_id)
                )
            )
        payload = _entity_node_payload(entity)
        payload["assets"] = [_asset_payload(asset) for asset in assets]
        return payload

    async def list_entity_relations(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        entity_id: str,
    ) -> list[dict[str, Any]]:
        """List all incoming and outgoing relations for one graph Entity."""

        async with self._database.session() as session:
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            entity_exists = await session.scalar(
                select(LogicalEntity.entity_id).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == entity_id,
                )
            )
            if entity_exists is None:
                raise ValueError("entity does not exist in graph")
            relations = list(
                await session.scalars(
                    select(LogicalEntityRelation)
                    .where(
                        LogicalEntityRelation.graph_id == graph_id,
                        or_(
                            LogicalEntityRelation.source_entity_id == entity_id,
                            LogicalEntityRelation.target_entity_id == entity_id,
                        ),
                    )
                    .order_by(LogicalEntityRelation.relation_id)
                )
            )
        return [_relation_payload(relation) for relation in relations]

    async def create_entity(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        name: str,
        entity_type: str = "",
        semantic: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        """Create a graph-local logical Entity."""

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            entity = LogicalEntity(
                graph_id=graph_id,
                entity_id=id_factory("entity")(),
                name=name,
                entity_type=entity_type,
                semantic=semantic,
                description=description,
            )
            session.add(entity)
            await session.flush()
        return _entity_node_payload(entity)

    async def delete_entity(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        entity_id: str,
    ) -> dict[str, Any]:
        """Delete one graph Entity and its bound Assets/relations."""

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            entity = await session.scalar(
                select(LogicalEntity).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == entity_id,
                )
            )
            if entity is None:
                raise ValueError("entity does not exist in graph")
            await session.delete(entity)
        return {"deleted_entity_id": entity_id}

    async def move_asset_to_entity(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        asset_id: str,
        entity_id: str,
        binding_role: str = "reference",
        description: str = "",
    ) -> dict[str, Any]:
        """Move a graph Asset to exactly one target Entity."""

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            asset_in_graph = await session.scalar(
                select(GraphAsset.asset_id).where(
                    GraphAsset.graph_id == graph_id,
                    GraphAsset.workspace_id == workspace_id,
                    GraphAsset.asset_id == asset_id,
                )
            )
            if asset_in_graph is None:
                raise ValueError("asset does not belong to graph")
            entity_exists = await session.scalar(
                select(LogicalEntity.entity_id).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == entity_id,
                )
            )
            if entity_exists is None:
                raise ValueError("entity does not exist in graph")
            existing_bindings = list(
                await session.scalars(
                    select(GraphAssetBinding).where(
                        GraphAssetBinding.graph_id == graph_id,
                        GraphAssetBinding.asset_id == asset_id,
                    )
                )
            )
            for binding in existing_bindings:
                await session.delete(binding)
            await session.flush()
            session.add(
                GraphAssetBinding(
                    graph_id=graph_id,
                    asset_id=asset_id,
                    entity_id=entity_id,
                    binding_role=binding_role,
                    description=description,
                )
            )
        return {"asset_id": asset_id, "entity_id": entity_id, "moved": True}

    async def merge_entities(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        target_entity_id: str,
        source_entity_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Merge source Entities into an existing target Entity."""

        source_ids = _deduplicated_strings(source_entity_ids)
        if not source_ids:
            raise ValueError("at least one source entity is required")
        if target_entity_id in source_ids:
            raise ValueError("target entity cannot also be a source entity")
        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            entities = list(
                await session.scalars(
                    select(LogicalEntity).where(
                        LogicalEntity.graph_id == graph_id,
                        LogicalEntity.entity_id.in_([target_entity_id, *source_ids]),
                    )
                )
            )
            entities_by_id = {entity.entity_id: entity for entity in entities}
            missing = [
                entity_id
                for entity_id in [target_entity_id, *source_ids]
                if entity_id not in entities_by_id
            ]
            if missing:
                raise ValueError(f"entities do not exist in graph: {', '.join(missing)}")

            source_bindings = list(
                await session.scalars(
                    select(GraphAssetBinding).where(
                        GraphAssetBinding.graph_id == graph_id,
                        GraphAssetBinding.entity_id.in_(source_ids),
                    )
                )
            )
            target_asset_ids = set(
                await session.scalars(
                    select(GraphAssetBinding.asset_id).where(
                        GraphAssetBinding.graph_id == graph_id,
                        GraphAssetBinding.entity_id == target_entity_id,
                    )
                )
            )
            for binding in source_bindings:
                if binding.asset_id not in target_asset_ids:
                    session.add(
                        GraphAssetBinding(
                            graph_id=graph_id,
                            asset_id=binding.asset_id,
                            entity_id=target_entity_id,
                            binding_role=binding.binding_role,
                            description=binding.description,
                        )
                    )
                    target_asset_ids.add(binding.asset_id)
                await session.delete(binding)
            await session.flush()

            source_relations = list(
                await session.scalars(
                    select(LogicalEntityRelation).where(
                        LogicalEntityRelation.graph_id == graph_id,
                        or_(
                            LogicalEntityRelation.source_entity_id.in_(source_ids),
                            LogicalEntityRelation.target_entity_id.in_(source_ids),
                        ),
                    )
                )
            )
            preserved_keys = set(
                (
                    row.source_entity_id,
                    row.target_entity_id,
                    row.relation_type,
                )
                for row in (
                    await session.execute(
                        select(
                            LogicalEntityRelation.source_entity_id,
                            LogicalEntityRelation.target_entity_id,
                            LogicalEntityRelation.relation_type,
                        ).where(
                            LogicalEntityRelation.graph_id == graph_id,
                            ~or_(
                                LogicalEntityRelation.source_entity_id.in_(source_ids),
                                LogicalEntityRelation.target_entity_id.in_(source_ids),
                            ),
                        )
                    )
                ).all()
            )
            for relation in source_relations:
                source_id = (
                    target_entity_id
                    if relation.source_entity_id in source_ids
                    else relation.source_entity_id
                )
                target_id = (
                    target_entity_id
                    if relation.target_entity_id in source_ids
                    else relation.target_entity_id
                )
                key = (source_id, target_id, relation.relation_type)
                if source_id != target_id and key not in preserved_keys:
                    session.add(
                        LogicalEntityRelation(
                            graph_id=graph_id,
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            relation_type=relation.relation_type,
                            description=relation.description,
                        )
                    )
                    preserved_keys.add(key)
                await session.delete(relation)
            await session.flush()
            for entity_id in source_ids:
                await session.delete(entities_by_id[entity_id])
        return {
            "target_entity_id": target_entity_id,
            "merged_entity_ids": source_ids,
            "moved_asset_count": len({binding.asset_id for binding in source_bindings}),
        }

    async def split_entity(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        entity_id: str,
        parts: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Split one Entity and assign each bound Asset to a new Entity."""

        if len(parts) < 2:
            raise ValueError("split requires at least two new entities")
        part_asset_ids = [
            asset_id
            for part in parts
            for asset_id in _deduplicated_strings(part.get("asset_ids", []))
        ]
        if len(part_asset_ids) != len(set(part_asset_ids)):
            raise ValueError("each asset can be assigned to only one split entity")

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            source = await session.scalar(
                select(LogicalEntity).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == entity_id,
                )
            )
            if source is None:
                raise ValueError("entity does not exist in graph")
            source_bindings = list(
                await session.scalars(
                    select(GraphAssetBinding).where(
                        GraphAssetBinding.graph_id == graph_id,
                        GraphAssetBinding.entity_id == entity_id,
                    )
                )
            )
            source_asset_ids = {binding.asset_id for binding in source_bindings}
            if source_asset_ids != set(part_asset_ids):
                missing = sorted(source_asset_ids - set(part_asset_ids))
                unknown = sorted(set(part_asset_ids) - source_asset_ids)
                details = []
                if missing:
                    details.append(f"unassigned assets: {', '.join(missing)}")
                if unknown:
                    details.append(f"assets not bound to source: {', '.join(unknown)}")
                raise ValueError("invalid split asset assignment (" + "; ".join(details) + ")")

            source_relations = list(
                await session.scalars(
                    select(LogicalEntityRelation).where(
                        LogicalEntityRelation.graph_id == graph_id,
                        or_(
                            LogicalEntityRelation.source_entity_id == entity_id,
                            LogicalEntityRelation.target_entity_id == entity_id,
                        ),
                    )
                )
            )
            created: list[LogicalEntity] = []
            for part in parts:
                created.append(
                    LogicalEntity(
                        graph_id=graph_id,
                        entity_id=id_factory("entity")(),
                        name=str(part["name"]),
                        entity_type=str(part.get("entity_type", "")),
                        semantic=str(part.get("semantic", "")),
                        description=str(part.get("description", "")),
                    )
                )
            session.add_all(created)
            await session.flush()
            part_by_asset = {
                asset_id: created[index].entity_id
                for index, part in enumerate(parts)
                for asset_id in _deduplicated_strings(part.get("asset_ids", []))
            }
            for binding in source_bindings:
                target_id = part_by_asset[binding.asset_id]
                session.add(
                    GraphAssetBinding(
                        graph_id=graph_id,
                        asset_id=binding.asset_id,
                        entity_id=target_id,
                        binding_role=binding.binding_role,
                        description=binding.description,
                    )
                )
                await session.delete(binding)
            await session.flush()
            # Preserve existing narrative edges on the first new part until
            # relation-specific reassignment is added to the tool contract.
            relation_owner = created[0].entity_id
            replacement_relations: list[tuple[str, str, str, str]] = []
            for relation in source_relations:
                source_id = (
                    relation_owner
                    if relation.source_entity_id == entity_id
                    else relation.source_entity_id
                )
                target_id = (
                    relation_owner
                    if relation.target_entity_id == entity_id
                    else relation.target_entity_id
                )
                await session.delete(relation)
                if source_id != target_id:
                    replacement_relations.append(
                        (source_id, target_id, relation.relation_type, relation.description)
                    )
            await session.flush()
            for source_id, target_id, relation_type, description in replacement_relations:
                session.add(
                    LogicalEntityRelation(
                        graph_id=graph_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        relation_type=relation_type,
                        description=description,
                    )
                )
            await session.flush()
            await session.delete(source)
        return {
            "deleted_entity_id": entity_id,
            "created_entities": [_entity_node_payload(entity) for entity in created],
            "reassigned_relation_count": len(source_relations),
        }

    async def create_parent_relation(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        child_entity_id: str,
        parent_name: str,
        parent_entity_type: str = "",
        parent_semantic: str = "",
        parent_description: str = "",
        relation_description: str = "",
    ) -> dict[str, Any]:
        """Create a parent Entity and attach one child with a hierarchy edge."""

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            child = await session.scalar(
                select(LogicalEntity).where(
                    LogicalEntity.graph_id == graph_id,
                    LogicalEntity.entity_id == child_entity_id,
                )
            )
            if child is None:
                raise ValueError("child entity does not exist in graph")
            parent = LogicalEntity(
                graph_id=graph_id,
                entity_id=id_factory("entity")(),
                name=parent_name,
                entity_type=parent_entity_type,
                semantic=parent_semantic,
                description=parent_description,
            )
            session.add(parent)
            session.add(
                LogicalEntityRelation(
                    graph_id=graph_id,
                    source_entity_id=child_entity_id,
                    target_entity_id=parent.entity_id,
                    relation_type="hierarchy",
                    description=relation_description,
                )
            )
            await session.flush()
        return {
            "parent": _entity_node_payload(parent),
            "child_entity_id": child_entity_id,
            "relation_type": "hierarchy",
        }

    async def remove_relation(
        self,
        *,
        workspace_id: str,
        graph_id: str,
        relation_id: str,
    ) -> dict[str, Any]:
        """Delete one graph relation by its stable identifier."""

        async with self._database.session() as session, session.begin():
            await _require_narrative_graph(
                session, workspace_id=workspace_id, graph_id=graph_id
            )
            relation = await session.scalar(
                select(LogicalEntityRelation).where(
                    LogicalEntityRelation.graph_id == graph_id,
                    LogicalEntityRelation.relation_id == relation_id,
                )
            )
            if relation is None:
                raise ValueError("relation does not exist in graph")
            await session.delete(relation)
        return {"deleted_relation_id": relation_id}

async def _require_narrative_graph(
    session: AsyncSession,
    *,
    workspace_id: str,
    graph_id: str,
) -> None:
    graph = await session.scalar(
        select(NarrativeGraph.graph_id).where(
            NarrativeGraph.graph_id == graph_id,
            NarrativeGraph.workspace_id == workspace_id,
        )
    )
    if graph is None:
        raise ValueError("narrative graph does not exist in workspace")


def _deduplicated_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("Entity identifiers must be non-empty strings")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _entity_node_payload(entity: LogicalEntity) -> dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "semantic": entity.semantic,
        "description": entity.description,
    }


def _asset_payload(asset: Asset) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "file_name": asset.file_name,
        "asset_name": asset.asset_name,
        "asset_type": asset.asset_type,
        "asset_description": asset.asset_description,
        "relative_path": asset.source_locator.get("relative_path", ""),
    }


def _relation_payload(relation: LogicalEntityRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "source_entity_id": relation.source_entity_id,
        "target_entity_id": relation.target_entity_id,
        "relation_type": relation.relation_type,
        "description": relation.description,
    }


@dataclass(slots=True, frozen=True)
class IndexedEmbeddingAsset:
    """Indexed vector metadata used to materialize fused search vectors."""

    embedding_id: str
    asset_id: str
    source_file_id: str
    asset_type: str
    asset_name: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    file_tree_context: list[str]
    source_relative_path: str = ""
@dataclass(slots=True, frozen=True)
class PreparedSourceFile:
    source_file_id: str
    already_processed: bool
    asset_count: int = 0
    generation: int = 0


@dataclass(slots=True, frozen=True)
class PreparedVideoTaskSubmission:
    """Database outcome of creating one durable whole-video submission.

    The parent job, source-generation claim, and optional queued task are all
    committed together.  A reused completed source has no new task because its
    new parent job is terminal in that same transaction.
    """

    job_id: str
    source_file_id: str
    generation: int
    task_id: str | None
    already_processed: bool


@dataclass(slots=True, frozen=True)
class PreparedProcessingTaskSubmission:
    """Atomic image/text CPU task submission and its trusted route identity."""

    job_id: str
    source_file_id: str
    generation: int
    task_id: str | None
    task_kind: str
    resource_class: str
    route_key: str
    processor_version: int
    already_processed: bool


@dataclass(slots=True, frozen=True)
class PreparedJobProcessingTask:
    """One file attached to an existing multi-file browser import job."""

    job_id: str
    source_file_id: str
    generation: int
    task_id: str | None
    task_kind: str
    resource_class: str
    route_key: str
    processor_version: int
    already_processed: bool
    source_uri: str = ""


@dataclass(slots=True, frozen=True)
class ProcessingTaskSourceInput:
    source_file: DiscoveredFile
    source_uri: str
    sha256: str
    mime_type: str
    processing_fingerprint: str
    source_contexts: tuple[dict[str, Any], ...] = ()


class StaleAssetGenerationError(ValueError):
    """A delayed queue delivery belongs to an older source processing run."""


class LibraryClearBusyError(ValueError):
    """The asset library cannot be cleared while an import is still active."""


@dataclass(slots=True, frozen=True)
class LibraryClearSnapshot:
    """Durable records collected before every workspace is deleted."""

    workspace_count: int
    asset_count: int
    source_file_count: int
    embedding_count: int
    job_count: int


@dataclass(slots=True, frozen=True)
class WorkspaceDeleteSnapshot:
    workspace_id: str
    asset_count: int
    source_file_count: int
    embedding_count: int
    job_count: int
    storage_uris: tuple[str, ...]
    object_keys: tuple[str, ...]
    staging_paths: tuple[str, ...]


class AssetRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def list_workspaces(self) -> list[WorkspaceRecord]:
        async with self._database.session() as session:
            rows = list(await session.scalars(select(Workspace).order_by(Workspace.created_at)))
        return [
            WorkspaceRecord(
                workspace_id=row.workspace_id,
                name=row.name,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    async def create_workspace(
        self,
        *,
        name: str,
        workspace_id: str | None = None,
    ) -> WorkspaceRecord:
        async with self._database.session() as session, session.begin():
            workspace = Workspace(name=name)
            if workspace_id is not None:
                workspace.workspace_id = workspace_id
            session.add(workspace)
            await session.flush()
            await session.refresh(workspace)
            return WorkspaceRecord(
                workspace_id=workspace.workspace_id,
                name=workspace.name,
                created_at=workspace.created_at,
                updated_at=workspace.updated_at,
            )

    async def delete_workspace_records(self, *, workspace_id: str) -> WorkspaceDeleteSnapshot:
        """Delete exactly one workspace and collect its external cleanup targets."""
        async with self._database.session() as session, session.begin():
            workspace = await session.get(Workspace, workspace_id, with_for_update=True)
            if workspace is None:
                raise ValueError(f"workspace does not exist: {workspace_id}")

            source_rows = list(
                await session.execute(
                    select(SourceFile.storage_uri).where(SourceFile.workspace_id == workspace_id)
                )
            )
            asset_rows = list(
                await session.execute(
                    select(
                        Asset.derived_file_uri,
                        Asset.preview_uri,
                        Asset.file_info,
                        Asset.source_locator,
                        Asset.source_contexts,
                    ).where(Asset.workspace_id == workspace_id)
                )
            )
            object_keys = tuple(
                str(value)
                for value in await session.scalars(
                    select(QueryImageUpload.object_key).where(
                        QueryImageUpload.workspace_id == workspace_id
                    )
                )
            )
            staging_paths = tuple(
                str(value)
                for value in await session.scalars(
                    select(ProcessingJob.input_path).where(
                        ProcessingJob.workspace_id == workspace_id
                    )
                )
            )
            storage_uris: set[str] = {
                str(row.storage_uri) for row in source_rows if row.storage_uri
            }
            for row in asset_rows:
                _collect_storage_uris(row.derived_file_uri, storage_uris)
                _collect_storage_uris(row.preview_uri, storage_uris)
                _collect_storage_uris(row.file_info, storage_uris)
                _collect_storage_uris(row.source_locator, storage_uris)
                _collect_storage_uris(row.source_contexts, storage_uris)

            asset_count = int(
                await session.scalar(
                    select(func.count(Asset.asset_id)).where(Asset.workspace_id == workspace_id)
                )
                or 0
            )
            source_file_count = int(
                await session.scalar(
                    select(func.count(SourceFile.source_file_id)).where(
                        SourceFile.workspace_id == workspace_id
                    )
                )
                or 0
            )
            embedding_count = int(
                await session.scalar(
                    select(func.count(EmbeddingRecord.embedding_id)).where(
                        EmbeddingRecord.workspace_id == workspace_id
                    )
                )
                or 0
            )
            job_count = int(
                await session.scalar(
                    select(func.count(ProcessingJob.job_id)).where(
                        ProcessingJob.workspace_id == workspace_id
                    )
                )
                or 0
            )
            await session.delete(workspace)
            return WorkspaceDeleteSnapshot(
                workspace_id=workspace_id,
                asset_count=asset_count,
                source_file_count=source_file_count,
                embedding_count=embedding_count,
                job_count=job_count,
                storage_uris=tuple(sorted(storage_uris)),
                object_keys=object_keys,
                staging_paths=staging_paths,
            )

    async def create_job(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        total_count: int,
    ) -> str:
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(_resolve_path(input_path)),
                total_count=total_count,
                status=JobStatus.RUNNING.value,
                current_stage=PipelineStage.PARSING.value,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.flush()
            return job.job_id

    async def create_pending_import_job(
        self,
        *,
        workspace_id: str,
        import_root: Path,
    ) -> str:
        """Create the durable upload session before browser files are transferred."""
        root = _resolve_path(import_root)
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(root),
                total_count=0,
                status=JobStatus.QUEUED.value,
                current_stage=PipelineStage.DISCOVERING.value,
            )
            session.add(job)
            await session.flush()
            job.input_path = str(root / job.job_id)
            return job.job_id

    async def clear_all_records(self) -> LibraryClearSnapshot:
        """Clear every workspace and its owned records from PostgreSQL.

        PostgreSQL cascades remove all workspace-owned assets, jobs, Embeddings,
        Search history and query-image metadata. Orphaned model
        call logs are deleted explicitly. External cleanup targets are returned
        for the caller to remove after this transaction.
        """
        active_statuses = (
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.RETRYING.value,
        )
        async with self._database.session() as session, session.begin():
            workspaces = list(await session.scalars(select(Workspace).with_for_update()))

            active_job_count = int(
                await session.scalar(
                    select(func.count(ProcessingJob.job_id)).where(
                        ProcessingJob.status.in_(active_statuses)
                    )
                )
                or 0
            )
            if active_job_count:
                raise LibraryClearBusyError(
                    f"asset library has {active_job_count} active import job(s)"
                )

            asset_count = int(await session.scalar(select(func.count(Asset.asset_id))) or 0)
            source_file_count = int(
                await session.scalar(select(func.count(SourceFile.source_file_id))) or 0
            )
            job_count = int(await session.scalar(select(func.count(ProcessingJob.job_id))) or 0)
            embedding_count = int(
                await session.scalar(select(func.count(EmbeddingRecord.embedding_id))) or 0
            )
            await session.execute(delete(ModelCallLog))
            for workspace in workspaces:
                await session.delete(workspace)

            return LibraryClearSnapshot(
                workspace_count=len(workspaces),
                asset_count=asset_count,
                source_file_count=source_file_count,
                embedding_count=embedding_count,
                job_count=job_count,
            )

    async def start_import_job(
        self,
        *,
        job_id: str,
        total_count: int,
        post_asset_action: str = "none",
    ) -> None:
        """Freeze an upload session and make it available to ``PipelineRunner``."""
        if total_count < 1:
            raise ValueError("an import job must contain at least one file")
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status != JobStatus.QUEUED.value:
                raise ValueError(f"processing job cannot be started from status {job.status}")
            job.total_count = total_count
            job.status = JobStatus.RUNNING.value
            job.current_stage = PipelineStage.PARSING.value
            job.started_at = datetime.now(UTC)
            if post_asset_action not in {"none", "enrich"}:
                raise ValueError("post_asset_action must be none or enrich")
            job.post_asset_action = post_asset_action

    async def mark_import_dispatch_complete(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.dispatch_completed_at = datetime.now(UTC)

    async def claim_ready_import_workflow(
        self,
        *,
        worker_id: str,
        lease_seconds: float = 60.0,
    ) -> tuple[str, str, str] | None:
        if not worker_id or lease_seconds <= 0:
            raise ValueError("workflow worker and positive lease are required")
        now = datetime.now(UTC)
        async with self._database.session() as session, session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.post_asset_action == "enrich",
                    ProcessingJob.dispatch_completed_at.is_not(None),
                    ProcessingJob.assetization_completed_at.is_not(None),
                    ProcessingJob.status == JobStatus.RUNNING.value,
                    ProcessingJob.current_stage == PipelineStage.ASSET_STORED.value,
                    or_(
                        ProcessingJob.workflow_owner_id.is_(None),
                        ProcessingJob.workflow_lease_deadline_at <= now,
                    ),
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if job is None:
                return None
            token = uuid4().hex
            job.workflow_owner_id = worker_id
            job.workflow_lease_token = token
            job.workflow_lease_deadline_at = now + timedelta(seconds=lease_seconds)
            return job.job_id, job.workspace_id, token

    async def heartbeat_import_workflow(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        lease_seconds: float = 60.0,
    ) -> bool:
        now = datetime.now(UTC)
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if (
                job is None
                or job.workflow_owner_id != worker_id
                or job.workflow_lease_token != lease_token
                or job.workflow_lease_deadline_at is None
                or job.workflow_lease_deadline_at <= now
            ):
                return False
            job.workflow_lease_deadline_at = now + timedelta(seconds=lease_seconds)
            return True

    async def release_import_workflow(
        self,
        *,
        job_id: str,
        worker_id: str,
        lease_token: str,
        error: str,
    ) -> bool:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if (
                job is None
                or job.workflow_owner_id != worker_id
                or job.workflow_lease_token != lease_token
            ):
                return False
            job.current_stage = PipelineStage.ASSET_STORED.value
            job.workflow_owner_id = None
            job.workflow_lease_token = None
            job.workflow_lease_deadline_at = None
            job.error_info = [
                *job.error_info,
                {"stage": "enrichment_workflow", "error": error[:2000]},
            ]
            return True

    async def list_import_job_asset_ids(self, *, job_id: str) -> list[str]:
        from capsule.db.video_tasks import VideoProcessingTask

        async with self._database.session() as session:
            return list(
                await session.scalars(
                    select(Asset.asset_id)
                    .join(
                        VideoProcessingTask,
                        (VideoProcessingTask.source_file_id == Asset.source_file_id)
                        & (VideoProcessingTask.source_generation == Asset.generation),
                    )
                    .where(
                        VideoProcessingTask.parent_job_id == job_id,
                        VideoProcessingTask.status.in_(("result_committed", "completed")),
                        Asset.index_role != AssetIndexRole.PARENT.value,
                    )
                    .order_by(Asset.asset_id)
                )
            )

    async def mark_import_upload_activity(
        self,
        *,
        job_id: str,
        workspace_id: str,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
                .with_for_update()
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status != JobStatus.QUEUED.value:
                raise ValueError(f"processing job cannot accept uploads from status {job.status}")
            job.updated_at = datetime.now(UTC)

    async def get_or_create_source_file(
        self,
        *,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
    ) -> str:
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            source = await self._find_source_file(
                session,
                workspace_id=workspace_id,
                relative_path=source_file.relative_path,
                lock=True,
            )
            source_path = _resolve_path(Path(source_file.path))
            values = {
                "original_file_name": source_path.name,
                "file_type": source_file.extension,
                "mime_type": mime_type,
                "relative_path": source_file.relative_path,
                "file_tree_context": list(Path(source_file.relative_path).parent.parts)
                if Path(source_file.relative_path).parent != Path(".")
                else [],
                "storage_uri": source_path.as_uri(),
                "sha256": sha256,
                "file_size_bytes": source_file.size_bytes,
                "processing_status": ProcessingStatus.PROCESSING.value,
                "error_message": None,
            }
            if source is None:
                source = SourceFile(workspace_id=workspace_id, **values)
                session.add(source)
            else:
                for field, value in values.items():
                    setattr(source, field, value)
            await session.flush()
            return source.source_file_id

    async def prepare_source_file(
        self,
        *,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
    ) -> "PreparedSourceFile":
        """Claim a logical source or reuse its completed, byte-identical assets."""
        async with self._database.session() as session, session.begin():
            return await self._prepare_source_file_in_session(
                session,
                workspace_id=workspace_id,
                source_file=source_file,
                sha256=sha256,
                mime_type=mime_type,
                processing_fingerprint=processing_fingerprint,
            )

    async def create_video_task_submission(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        result_version: int = 1,
    ) -> PreparedVideoTaskSubmission:
        """Commit a durable video submission before its Redis delivery exists.

        Redis can be unavailable immediately after this returns: the scheduler
        republishes the queued task.  Conversely, a database rollback leaves no
        orphaned running job or claimed source generation behind.
        """
        if result_version < 1:
            raise ValueError("video task result_version must be positive")
        # Keep this import local: importing the task runtime loads the pipeline
        # package, whose historical exports depend on this repository module.
        from capsule.db.video_tasks import VideoProcessingTask

        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(_resolve_path(input_path)),
                total_count=1,
                status=JobStatus.RUNNING.value,
                current_stage=PipelineStage.PARSING.value,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.flush()

            prepared = await self._prepare_source_file_in_session(
                session,
                workspace_id=workspace_id,
                source_file=source_file,
                sha256=sha256,
                mime_type=mime_type,
                processing_fingerprint=processing_fingerprint,
            )
            if prepared.already_processed:
                job.completed_count = 1
                job.status = JobStatus.COMPLETED.value
                job.current_stage = PipelineStage.COMPLETED.value
                job.completed_at = datetime.now(UTC)
                return PreparedVideoTaskSubmission(
                    job_id=job.job_id,
                    source_file_id=prepared.source_file_id,
                    generation=prepared.generation,
                    task_id=None,
                    already_processed=True,
                )

            task = VideoProcessingTask(
                parent_job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                source_generation=prepared.generation,
                result_version=result_version,
                status="queued",
                stage="queued",
                attempt=0,
                progress={},
            )
            session.add(task)
            await session.flush()
            return PreparedVideoTaskSubmission(
                job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                generation=prepared.generation,
                task_id=task.task_id,
                already_processed=False,
            )

    async def create_cpu_processing_task_submission(
        self,
        *,
        workspace_id: str,
        input_path: Path,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        result_version: int = 1,
        processor_version: int = 1,
    ) -> PreparedProcessingTaskSubmission:
        """Atomically create a CPU image/text job, source generation and task.

        The source extension selects its trusted kind; callers cannot provide a
        route, worker pool, or arbitrary processor implementation.
        """
        if result_version < 1 or processor_version < 1:
            raise ValueError("processing task result and processor versions must be positive")
        from capsule.db.processing_task_persistence import cpu_contract_for_extension
        from capsule.db.video_tasks import VideoProcessingTask

        contract = cpu_contract_for_extension(source_file.extension)
        async with self._database.session() as session, session.begin():
            await self._ensure_workspace(session, workspace_id)
            job = ProcessingJob(
                workspace_id=workspace_id,
                input_path=str(_resolve_path(input_path)),
                total_count=1,
                status=JobStatus.RUNNING.value,
                current_stage=PipelineStage.PARSING.value,
                started_at=datetime.now(UTC),
            )
            session.add(job)
            await session.flush()
            prepared = await self._prepare_source_file_in_session(
                session,
                workspace_id=workspace_id,
                source_file=source_file,
                sha256=sha256,
                mime_type=mime_type,
                processing_fingerprint=processing_fingerprint,
            )
            if prepared.already_processed:
                job.completed_count = 1
                job.status = JobStatus.COMPLETED.value
                job.current_stage = PipelineStage.COMPLETED.value
                job.completed_at = datetime.now(UTC)
                return PreparedProcessingTaskSubmission(
                    job_id=job.job_id,
                    source_file_id=prepared.source_file_id,
                    generation=prepared.generation,
                    task_id=None,
                    task_kind=contract.task_kind.value,
                    resource_class=contract.resource_class.value,
                    route_key=contract.route_key,
                    processor_version=processor_version,
                    already_processed=True,
                )
            task = VideoProcessingTask(
                parent_job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                source_generation=prepared.generation,
                result_version=result_version,
                task_kind=contract.task_kind.value,
                resource_class=contract.resource_class.value,
                route_key=contract.route_key,
                processor_version=processor_version,
                status="queued",
                stage="queued",
                attempt=0,
                progress={},
            )
            session.add(task)
            await session.flush()
            return PreparedProcessingTaskSubmission(
                job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                generation=prepared.generation,
                task_id=task.task_id,
                task_kind=contract.task_kind.value,
                resource_class=contract.resource_class.value,
                route_key=contract.route_key,
                processor_version=processor_version,
                already_processed=False,
            )

    async def create_job_processing_task(
        self,
        *,
        job_id: str,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        source_contexts: list[dict[str, Any]] | None = None,
        result_version: int = 1,
        processor_version: int = 1,
    ) -> PreparedJobProcessingTask:
        """Attach one trusted file task to an existing browser import job.

        The browser job remains the only parent visible to the frontend. Every
        file task accounts exactly one of its ``total_count`` slots.
        """
        if result_version < 1 or processor_version < 1:
            raise ValueError("processing task result and processor versions must be positive")
        from capsule.db.processing_task_persistence import cpu_contract_for_extension
        from capsule.db.video_tasks import VideoProcessingTask
        from capsule.pipeline.video_task_runtime import ProcessingTaskKind, ResourceClass

        extension = source_file.extension.lower()
        if extension in {".mp4", ".mov"}:
            task_kind = ProcessingTaskKind.VIDEO.value
            resource_class = ResourceClass.MPS_VIDEO.value
            route_key = "mps_video"
        elif extension in {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}:
            task_kind = ProcessingTaskKind.AUDIO.value
            resource_class = ResourceClass.MPS_VIDEO.value
            route_key = "mps_video"
        else:
            contract = cpu_contract_for_extension(extension)
            task_kind = contract.task_kind.value
            resource_class = contract.resource_class.value
            route_key = contract.route_key

        async with self._database.session() as session, session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
                .with_for_update()
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status != JobStatus.RUNNING.value:
                raise ValueError(f"processing job cannot accept tasks from status {job.status}")
            if job.completed_count + job.failed_count >= job.total_count:
                raise ValueError("processing job already accounted every source file")

            prepared = await self._prepare_source_file_in_session(
                session,
                workspace_id=workspace_id,
                source_file=source_file,
                sha256=sha256,
                mime_type=mime_type,
                processing_fingerprint=processing_fingerprint,
            )
            if prepared.already_processed:
                job.completed_count += 1
                if job.completed_count + job.failed_count == job.total_count:
                    job.status = JobStatus.COMPLETED.value
                    job.current_stage = PipelineStage.COMPLETED.value
                    job.completed_at = datetime.now(UTC)
                return PreparedJobProcessingTask(
                    job_id=job.job_id,
                    source_file_id=prepared.source_file_id,
                    generation=prepared.generation,
                    task_id=None,
                    task_kind=task_kind,
                    resource_class=resource_class,
                    route_key=route_key,
                    processor_version=processor_version,
                    already_processed=True,
                    source_uri=_resolve_path(Path(source_file.path)).as_uri(),
                )

            task = VideoProcessingTask(
                parent_job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                source_generation=prepared.generation,
                result_version=result_version,
                task_kind=task_kind,
                resource_class=resource_class,
                route_key=route_key,
                processor_version=processor_version,
                status="queued",
                stage="queued",
                attempt=0,
                progress={},
                input_payload={
                    "source_sha256": sha256,
                    "processing_fingerprint": processing_fingerprint,
                    "source_contexts": source_contexts or [],
                },
            )
            session.add(task)
            await session.flush()
            return PreparedJobProcessingTask(
                job_id=job.job_id,
                source_file_id=prepared.source_file_id,
                generation=prepared.generation,
                task_id=task.task_id,
                task_kind=task_kind,
                resource_class=resource_class,
                route_key=route_key,
                processor_version=processor_version,
                already_processed=False,
                source_uri=_resolve_path(Path(source_file.path)).as_uri(),
            )

    async def prepare_import_task_batch(
        self,
        *,
        job_id: str,
        workspace_id: str,
        items: Sequence[ProcessingTaskSourceInput],
        post_asset_action: str = "none",
    ) -> list[PreparedJobProcessingTask]:
        """Atomically start a browser job and register its complete task batch."""
        from capsule.db.processing_task_persistence import cpu_contract_for_extension
        from capsule.db.video_tasks import VideoProcessingTask
        from capsule.pipeline.video_task_runtime import ProcessingTaskKind, ResourceClass

        if not items:
            raise ValueError("browser processing task batch cannot be empty")
        if post_asset_action not in {"none", "enrich"}:
            raise ValueError("post_asset_action must be none or enrich")
        async with self._database.session() as session, session.begin():
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
                .with_for_update()
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.status not in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
                raise ValueError(f"processing job cannot accept tasks from status {job.status}")
            if job.status == JobStatus.RUNNING.value and len(items) != job.total_count:
                raise ValueError("browser task batch does not match job total_count")

            existing = list(
                await session.execute(
                    select(VideoProcessingTask, SourceFile.storage_uri)
                    .join(
                        SourceFile,
                        SourceFile.source_file_id == VideoProcessingTask.source_file_id,
                    )
                    .where(VideoProcessingTask.parent_job_id == job_id)
                    .order_by(VideoProcessingTask.task_id)
                )
            )
            if existing:
                if len(existing) != job.total_count:
                    raise RuntimeError("browser job contains an incomplete durable task batch")
                job.dispatch_completed_at = job.dispatch_completed_at or datetime.now(UTC)
                return [
                    PreparedJobProcessingTask(
                        job_id=job_id,
                        source_file_id=task.source_file_id,
                        generation=task.source_generation,
                        task_id=task.task_id,
                        task_kind=task.task_kind,
                        resource_class=task.resource_class,
                        route_key=task.route_key,
                        processor_version=task.processor_version,
                        already_processed=False,
                        source_uri=source_uri,
                    )
                    for task, source_uri in existing
                ]

            if job.status != JobStatus.QUEUED.value:
                raise RuntimeError("running browser job is missing its durable task batch")
            job.total_count = len(items)
            job.status = JobStatus.RUNNING.value
            job.current_stage = PipelineStage.PARSING.value
            job.started_at = datetime.now(UTC)
            job.post_asset_action = post_asset_action
            prepared_tasks: list[PreparedJobProcessingTask] = []
            for item in items:
                # The browser-side URI is only a convenience for the first
                # Redis delivery.  Persist and return the same canonical URI
                # as SourceFile so a forged/stale caller value cannot create
                # a task contract that workers subsequently reject.
                source_uri = _resolve_path(Path(item.source_file.path)).as_uri()
                extension = item.source_file.extension.lower()
                if extension in {".mp4", ".mov"}:
                    task_kind = ProcessingTaskKind.VIDEO.value
                    resource_class = ResourceClass.MPS_VIDEO.value
                    route_key = "mps_video"
                elif extension in {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg"}:
                    task_kind = ProcessingTaskKind.AUDIO.value
                    resource_class = ResourceClass.MPS_VIDEO.value
                    route_key = "mps_video"
                else:
                    contract = cpu_contract_for_extension(extension)
                    task_kind = contract.task_kind.value
                    resource_class = contract.resource_class.value
                    route_key = contract.route_key
                prepared = await self._prepare_source_file_in_session(
                    session,
                    workspace_id=workspace_id,
                    source_file=item.source_file,
                    sha256=item.sha256,
                    mime_type=item.mime_type,
                    processing_fingerprint=item.processing_fingerprint,
                    force_reprocess=True,
                )
                task = VideoProcessingTask(
                    parent_job_id=job_id,
                    source_file_id=prepared.source_file_id,
                    source_generation=prepared.generation,
                    result_version=1,
                    task_kind=task_kind,
                    resource_class=resource_class,
                    route_key=route_key,
                    processor_version=1,
                    status="queued",
                    stage="queued",
                    attempt=0,
                    progress={},
                    input_payload={
                        "source_sha256": item.sha256,
                        "source_uri": source_uri,
                        "processing_fingerprint": item.processing_fingerprint,
                        "source_contexts": list(item.source_contexts),
                    },
                )
                session.add(task)
                await session.flush()
                prepared_tasks.append(
                    PreparedJobProcessingTask(
                        job_id=job_id,
                        source_file_id=prepared.source_file_id,
                        generation=prepared.generation,
                        task_id=task.task_id,
                        task_kind=task_kind,
                        resource_class=resource_class,
                        route_key=route_key,
                        processor_version=1,
                        already_processed=False,
                        source_uri=source_uri,
                    )
                )
            job.dispatch_completed_at = datetime.now(UTC)
            return prepared_tasks

    async def replace_assets(
        self,
        *,
        source_file_id: str,
        assets: list[AssetCreate],
    ) -> StoredFileResult:
        if any(asset.source_file_id != source_file_id for asset in assets):
            raise ValueError("all assets must belong to the supplied source_file_id")
        _validate_asset_hierarchy(assets, require_batch_parent=True)

        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            generation = source.processing_generation
            if any(asset.generation != generation for asset in assets):
                raise ValueError("asset generation does not match the source generation")

            rows = await session.scalars(
                select(Asset).where(Asset.source_file_id == source_file_id).with_for_update()
            )
            existing = {asset.asset_key: asset for asset in rows}
            stored_by_key: dict[str, Asset] = {}

            def store(values: AssetCreate) -> Asset:
                current = existing.pop(values.asset_key, None)
                if current is None:
                    current = Asset(**_asset_values(values))
                    session.add(current)
                else:
                    content_changed = current.content_hash != values.content_hash
                    user_name = (
                        current.asset_name
                        if current.asset_name_source == AssetNameSource.USER.value
                        else None
                    )
                    _update_asset(current, values, content_changed=content_changed)
                    if user_name is not None:
                        current.asset_name = user_name
                        current.asset_name_source = AssetNameSource.USER.value
                stored_by_key[values.asset_key] = current
                return current

            # A child cannot be inserted until its persisted parent has a real ID:
            # the database check constraint deliberately disallows dangling children.
            for values in assets:
                if values.index_role != AssetIndexRole.CHILD:
                    store(values)

            await session.flush()
            for values in assets:
                if values.index_role != AssetIndexRole.CHILD:
                    continue
                current = store(values)
                _resolve_asset_parent(
                    asset=current,
                    values=values,
                    source_file_id=source_file_id,
                    possible_parents=stored_by_key,
                )

            await session.flush()

            for stale in existing.values():
                await session.delete(stale)

            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            await session.flush()
            return StoredFileResult(
                source_file_id=source_file_id,
                asset_ids=[stored_by_key[asset.asset_key].asset_id for asset in assets],
                indexable_asset_ids=[
                    stored_by_key[asset.asset_key].asset_id
                    for asset in assets
                    if asset.index_role != AssetIndexRole.PARENT
                ],
            )

    async def upsert_generated_asset(
        self,
        *,
        source_file_id: str,
        generation: int,
        asset: AssetCreate,
    ) -> str:
        """Store one completed Segment, rejecting deliveries from an obsolete run."""
        if asset.source_file_id != source_file_id or asset.generation != generation:
            raise ValueError("asset does not belong to the supplied source generation")
        _validate_asset_hierarchy([asset], require_batch_parent=False)
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            possible_parents: dict[str, Asset] = {}
            if asset.index_role == AssetIndexRole.CHILD:
                parent = await session.scalar(
                    select(Asset)
                    .where(
                        Asset.source_file_id == source_file_id,
                        Asset.asset_key == asset.parent_asset_key,
                    )
                    .with_for_update()
                )
                if parent is not None:
                    possible_parents[asset.parent_asset_key or ""] = parent
            current = await session.scalar(
                select(Asset)
                .where(
                    Asset.source_file_id == source_file_id,
                    Asset.asset_key == asset.asset_key,
                )
                .with_for_update()
            )
            if current is None:
                current = Asset(**_asset_values(asset))
                session.add(current)
            else:
                content_changed = current.content_hash != asset.content_hash
                user_name = (
                    current.asset_name
                    if current.asset_name_source == AssetNameSource.USER.value
                    else None
                )
                _update_asset(current, asset, content_changed=content_changed)
                if user_name is not None:
                    current.asset_name = user_name
                    current.asset_name_source = AssetNameSource.USER.value
            if asset.index_role == AssetIndexRole.CHILD:
                _resolve_asset_parent(
                    asset=current,
                    values=asset,
                    source_file_id=source_file_id,
                    possible_parents=possible_parents,
                )
            else:
                current.parent_asset_id = None
            await session.flush()
            return current.asset_id

    async def assert_current_generation(
        self,
        *,
        source_file_id: str,
        generation: int,
    ) -> None:
        """Reject stale queue work before it can overwrite a stable object key."""
        async with self._database.session() as session:
            current_generation = await session.scalar(
                select(SourceFile.processing_generation).where(
                    SourceFile.source_file_id == source_file_id
                )
            )
        if current_generation is None:
            raise ValueError(f"source file does not exist: {source_file_id}")
        if current_generation != generation:
            raise StaleAssetGenerationError(
                f"source generation advanced from {generation} to {current_generation}"
            )

    async def finalize_asset_generation(
        self,
        *,
        source_file_id: str,
        generation: int,
    ) -> StoredFileResult:
        """Publish one complete generation and remove stale Asset rows atomically."""
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            await session.execute(
                delete(Asset).where(
                    Asset.source_file_id == source_file_id,
                    Asset.generation != generation,
                )
            )
            stored_assets = list(
                await session.execute(
                    select(Asset.asset_id, Asset.index_role)
                    .where(
                        Asset.source_file_id == source_file_id,
                        Asset.generation == generation,
                    )
                    .order_by(Asset.asset_key)
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            asset_ids = [asset_id for asset_id, _ in stored_assets]
            return StoredFileResult(
                source_file_id=source_file_id,
                asset_ids=asset_ids,
                indexable_asset_ids=[
                    asset_id
                    for asset_id, index_role in stored_assets
                    if index_role != AssetIndexRole.PARENT.value
                ],
            )

    async def finalize_asset_generation_if_complete(
        self,
        *,
        source_file_id: str,
        generation: int,
        expected_asset_count: int,
    ) -> bool:
        """Finalize a recovered stream generation once every Segment is durable."""
        if expected_asset_count < 1:
            raise ValueError("expected_asset_count must be positive")
        async with self._database.session() as session, session.begin():
            source = await session.get(SourceFile, source_file_id, with_for_update=True)
            if source is None:
                raise ValueError(f"source file does not exist: {source_file_id}")
            if source.processing_generation != generation:
                raise StaleAssetGenerationError(
                    f"source generation advanced from {generation} "
                    f"to {source.processing_generation}"
                )
            stored_count = int(
                await session.scalar(
                    select(func.count(Asset.asset_id)).where(
                        Asset.source_file_id == source_file_id,
                        Asset.generation == generation,
                    )
                )
                or 0
            )
            if stored_count < expected_asset_count:
                return False
            if stored_count > expected_asset_count:
                raise ValueError("source generation contains more Assets than its upload manifest")
            await session.execute(
                delete(Asset).where(
                    Asset.source_file_id == source_file_id,
                    Asset.generation != generation,
                )
            )
            source.processing_status = ProcessingStatus.COMPLETED.value
            source.error_message = None
            return True

    async def record_file_failure(
        self,
        *,
        job_id: str,
        source_file_id: str | None,
        relative_path: str,
        error: str,
        generation: int | None = None,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.failed_count += 1
            job.error_info = [
                *job.error_info,
                {"relative_path": relative_path, "error": error[:2000]},
            ]
            if source_file_id is not None:
                source = await session.get(SourceFile, source_file_id, with_for_update=True)
                if source is not None and (
                    generation is None or source.processing_generation == generation
                ):
                    source.processing_status = ProcessingStatus.FAILED.value
                    source.error_message = error[:2000]

    async def record_file_success(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.completed_count += 1

    async def add_job_stage_durations(
        self,
        *,
        job_id: str,
        durations_ms: dict[str, float],
    ) -> None:
        """Accumulate measured wall-clock work for independently timed stages."""
        if not durations_ms:
            return
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            current = dict(job.stage_durations_ms)
            for stage, duration_ms in durations_ms.items():
                current[stage] = round(
                    current.get(stage, 0.0) + max(0.0, duration_ms),
                    3,
                )
            job.stage_durations_ms = current

    async def finalize_job(self, *, job_id: str) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if job.failed_count == 0:
                job.status = JobStatus.COMPLETED.value
                job.current_stage = PipelineStage.COMPLETED.value
            elif job.completed_count == 0:
                job.status = JobStatus.FAILED.value
                job.current_stage = PipelineStage.FAILED.value
            else:
                job.status = JobStatus.PARTIAL_FAILED.value
                job.current_stage = PipelineStage.COMPLETED.value
            job.completed_at = datetime.now(UTC)

    async def fail_job(self, *, job_id: str, error: str) -> None:
        """Fail an import before per-file handling could record an outcome."""
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.status = JobStatus.FAILED.value
            job.current_stage = PipelineStage.FAILED.value
            job.error_info = [*job.error_info, {"relative_path": "", "error": error[:2000]}]
            if job.failed_count == 0:
                job.failed_count = 1
            job.completed_at = datetime.now(UTC)

    async def get_job(self, *, job_id: str, workspace_id: str) -> ProcessingJobRecord:
        async with self._database.session() as session:
            job = await session.scalar(
                select(ProcessingJob).where(
                    ProcessingJob.job_id == job_id,
                    ProcessingJob.workspace_id == workspace_id,
                )
            )
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            return ProcessingJobRecord(
                job_id=job.job_id,
                workspace_id=job.workspace_id,
                input_path=job.input_path,
                total_count=job.total_count,
                completed_count=job.completed_count,
                failed_count=job.failed_count,
                status=job.status,
                current_stage=job.current_stage,
                error_info=list(job.error_info),
                stage_durations_ms=dict(job.stage_durations_ms),
                started_at=job.started_at,
                completed_at=job.completed_at,
            )

    async def list_jobs(
        self,
        *,
        workspace_id: str,
        limit: int = 50,
    ) -> list[ProcessingJobRecord]:
        async with self._database.session() as session:
            rows = await session.scalars(
                select(ProcessingJob)
                .where(ProcessingJob.workspace_id == workspace_id)
                .order_by(ProcessingJob.created_at.desc(), ProcessingJob.job_id.desc())
                .limit(limit)
            )
            return [_processing_job_record(job) for job in rows]

    async def clear_jobs(self, *, workspace_id: str) -> int:
        """Remove every processing-job record owned by one workspace."""
        async with self._database.session() as session, session.begin():
            deleted_ids = list(
                await session.scalars(
                    delete(ProcessingJob)
                    .where(ProcessingJob.workspace_id == workspace_id)
                    .returning(ProcessingJob.job_id)
                )
            )
            return len(deleted_ids)

    async def list_asset_views(
        self,
        *,
        workspace_id: str,
        asset_type: str | None = None,
        processing_status: str | None = None,
        source_file_id: str | None = None,
        query: str | None = None,
        asset_ids: Sequence[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AssetListResponse:
        filters = [
            Asset.workspace_id == workspace_id,
            Asset.generation == SourceFile.processing_generation,
        ]
        if asset_type:
            filters.append(Asset.asset_type == asset_type)
        if processing_status:
            filters.append(Asset.processing_status == processing_status)
        if source_file_id:
            filters.append(Asset.source_file_id == source_file_id)
        if asset_ids:
            filters.append(Asset.asset_id.in_(asset_ids))
        normalized_query = query.strip() if query else ""
        if normalized_query:
            pattern = f"%{normalized_query}%"
            filters.append(
                or_(
                    Asset.file_name.ilike(pattern),
                    Asset.asset_name.ilike(pattern),
                    Asset.asset_description.ilike(pattern),
                    SourceFile.relative_path.ilike(pattern),
                )
            )

        async with self._database.session() as session:
            total = int(
                await session.scalar(
                    select(func.count())
                    .select_from(Asset)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(*filters)
                )
                or 0
            )
            rows = (
                await session.execute(
                    select(Asset, SourceFile)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(*filters)
                    .order_by(Asset.created_at.desc(), Asset.asset_id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
            embeddings = await _latest_embedding_states(
                session,
                [asset.asset_id for asset, _ in rows],
            )
        return AssetListResponse(
            items=[
                _asset_view_record(
                    asset,
                    source,
                    embeddings=embeddings.get(asset.asset_id, []),
                )
                for asset, source in rows
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    async def get_asset_view(
        self,
        *,
        asset_id: str,
        workspace_id: str,
    ) -> AssetViewRecord:
        result = await self.list_asset_views(
            workspace_id=workspace_id,
            asset_ids=[asset_id],
            limit=1,
        )
        if not result.items:
            raise ValueError(f"asset does not exist: {asset_id}")
        return result.items[0]

    async def get_asset_media(
        self,
        *,
        asset_id: str,
        workspace_id: str,
    ) -> AssetMediaTarget:
        async with self._database.session() as session:
            row = (
                await session.execute(
                    select(Asset, SourceFile)
                    .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
                    .where(
                        Asset.asset_id == asset_id,
                        Asset.workspace_id == workspace_id,
                        Asset.generation == SourceFile.processing_generation,
                    )
                )
            ).one_or_none()
        if row is None:
            raise ValueError(f"asset does not exist: {asset_id}")
        asset, source = row
        return AssetMediaTarget(
            asset_id=asset.asset_id,
            workspace_id=asset.workspace_id,
            asset_type=asset.asset_type,
            source_storage_uri=source.storage_uri,
            source_mime_type=source.mime_type,
            preview_uri=asset.preview_uri,
            derived_file_uri=asset.derived_file_uri,
        )

    async def set_job_stage(
        self,
        *,
        job_id: str,
        stage: PipelineStage,
    ) -> None:
        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            job.status = JobStatus.RUNNING.value
            job.current_stage = stage.value
            job.completed_at = None

    async def begin_asset_enrichment(self, *, asset_ids: Sequence[str]) -> None:
        if not asset_ids:
            return
        async with self._database.session() as session, session.begin():
            rows = await session.scalars(
                select(Asset).where(Asset.asset_id.in_(asset_ids)).with_for_update()
            )
            for asset in rows:
                asset.processing_status = ProcessingStatus.PROCESSING.value
                asset.error_message = None

    async def store_understanding(
        self,
        *,
        asset_id: str,
        understanding: AssetUnderstanding,
        raw_content: str | None = None,
        file_info_updates: Mapping[str, Any] | None = None,
    ) -> None:
        async with self._database.session() as session, session.begin():
            asset = await session.get(Asset, asset_id, with_for_update=True)
            if asset is None:
                raise ValueError(f"asset does not exist: {asset_id}")
            features = understanding.features.model_dump(mode="json")
            semantic_changed = asset.asset_description not in {
                None,
                understanding.asset_description,
            } or (bool(asset.asset_features) and asset.asset_features != features)
            if raw_content is not None and asset.raw_content not in {None, raw_content}:
                semantic_changed = True
            if asset.asset_name_source != AssetNameSource.USER.value:
                asset.asset_name = understanding.asset_name
                asset.asset_name_source = AssetNameSource.MODEL.value
            asset.asset_description = understanding.asset_description
            asset.asset_features = features
            if raw_content is not None:
                asset.raw_content = raw_content
            if file_info_updates:
                asset.file_info = {**(asset.file_info or {}), **file_info_updates}
            asset.processing_status = ProcessingStatus.PROCESSING.value
            asset.error_message = None
            if semantic_changed:
                asset.feature_revision += 1
                asset.embedding_revision += 1

    async def finalize_enrichment(
        self,
        *,
        job_id: str,
        asset_ids: Sequence[str],
        errors: Sequence[dict[str, str]],
        workflow_lease_token: str | None = None,
    ) -> None:
        errors_by_asset: dict[str, list[str]] = {}
        for error in errors:
            asset_id = error.get("asset_id", "")
            if asset_id:
                errors_by_asset.setdefault(asset_id, []).append(error.get("error", "failed"))

        async with self._database.session() as session, session.begin():
            job = await session.get(ProcessingJob, job_id, with_for_update=True)
            if job is None:
                raise ValueError(f"processing job does not exist: {job_id}")
            if workflow_lease_token is not None and (
                job.workflow_lease_token != workflow_lease_token
                or job.workflow_lease_deadline_at is None
                or job.workflow_lease_deadline_at <= datetime.now(UTC)
            ):
                raise ValueError("import enrichment workflow lease was lost")
            rows = list(
                await session.scalars(
                    select(Asset).where(Asset.asset_id.in_(asset_ids)).with_for_update()
                )
            )
            for asset in rows:
                asset_errors = errors_by_asset.get(asset.asset_id, [])
                if asset_errors:
                    asset.processing_status = ProcessingStatus.PARTIAL_FAILED.value
                    asset.error_message = "; ".join(asset_errors)[:2000]
                else:
                    asset.processing_status = ProcessingStatus.COMPLETED.value
                    asset.error_message = None

            source_ids = {asset.source_file_id for asset in rows}
            for source_id in source_ids:
                source = await session.get(SourceFile, source_id, with_for_update=True)
                if source is None:
                    continue
                # Incremental video Assets can finish Understanding while later
                # Segments are still rendering/uploading. Only generation
                # finalization may publish the SourceFile in that window.
                if source.processing_status == ProcessingStatus.PROCESSING.value:
                    continue
                source_assets = list(
                    await session.scalars(
                        select(Asset).where(
                            Asset.source_file_id == source_id,
                            Asset.generation == source.processing_generation,
                        )
                    )
                )
                failed_assets = [
                    asset
                    for asset in source_assets
                    if asset.processing_status
                    in {
                        ProcessingStatus.FAILED.value,
                        ProcessingStatus.PARTIAL_FAILED.value,
                    }
                ]
                source.processing_status = (
                    ProcessingStatus.PARTIAL_FAILED.value
                    if failed_assets
                    else ProcessingStatus.COMPLETED.value
                )
                source.error_message = (
                    "; ".join(
                        asset.error_message or "asset enrichment failed" for asset in failed_assets
                    )[:2000]
                    if failed_assets
                    else None
                )

            if errors:
                job.status = JobStatus.PARTIAL_FAILED.value
                job.error_info = [
                    *job.error_info,
                    *[
                        {
                            "asset_id": error.get("asset_id", ""),
                            "stage": error.get("stage", ""),
                            "error": error.get("error", "")[:2000],
                        }
                        for error in errors
                    ],
                ]
            elif job.failed_count:
                job.status = JobStatus.PARTIAL_FAILED.value
            else:
                job.status = JobStatus.COMPLETED.value
            job.current_stage = PipelineStage.COMPLETED.value
            job.completed_at = datetime.now(UTC)
            job.workflow_owner_id = None
            job.workflow_lease_token = None
            job.workflow_lease_deadline_at = None

    async def _prepare_source_file_in_session(
        self,
        session: AsyncSession,
        *,
        workspace_id: str,
        source_file: DiscoveredFile,
        sha256: str,
        mime_type: str,
        processing_fingerprint: str,
        force_reprocess: bool = False,
    ) -> PreparedSourceFile:
        """Implement ``prepare_source_file`` inside an existing transaction."""
        await self._ensure_workspace(session, workspace_id)
        source = await self._find_source_file(
            session,
            workspace_id=workspace_id,
            relative_path=source_file.relative_path,
            lock=True,
        )
        source_path = _resolve_path(Path(source_file.path))
        metadata = {
            "original_file_name": source_path.name,
            "file_type": source_file.extension,
            "mime_type": mime_type,
            "relative_path": source_file.relative_path,
            "file_tree_context": list(Path(source_file.relative_path).parent.parts)
            if Path(source_file.relative_path).parent != Path(".")
            else [],
            "storage_uri": source_path.as_uri(),
            "file_size_bytes": source_file.size_bytes,
        }
        if (
            not force_reprocess
            and source is not None
            and source.sha256 == sha256
            and source.processing_fingerprint == processing_fingerprint
            and source.processing_status == ProcessingStatus.COMPLETED.value
        ):
            for field, value in metadata.items():
                setattr(source, field, value)
            asset_count = int(
                await session.scalar(
                    select(func.count(Asset.asset_id)).where(
                        Asset.source_file_id == source.source_file_id
                    )
                )
                or 0
            )
            return PreparedSourceFile(
                source_file_id=source.source_file_id,
                already_processed=True,
                asset_count=asset_count,
                generation=source.processing_generation,
            )

        values = {
            **metadata,
            "sha256": sha256,
            "processing_fingerprint": processing_fingerprint,
            "processing_status": ProcessingStatus.PROCESSING.value,
            "error_message": None,
        }
        if source is None:
            source = SourceFile(
                workspace_id=workspace_id,
                processing_generation=1,
                **values,
            )
            session.add(source)
        else:
            for field, value in values.items():
                setattr(source, field, value)
            source.processing_generation += 1
        await session.flush()
        return PreparedSourceFile(
            source_file_id=source.source_file_id,
            already_processed=False,
            generation=source.processing_generation,
        )

    @staticmethod
    async def _ensure_workspace(session: AsyncSession, workspace_id: str) -> None:
        if await session.get(Workspace, workspace_id) is None:
            session.add(Workspace(workspace_id=workspace_id, name=workspace_id))
            await session.flush()

    @staticmethod
    async def _find_source_file(
        session: AsyncSession,
        *,
        workspace_id: str,
        relative_path: str,
        lock: bool,
    ) -> SourceFile | None:
        statement = select(SourceFile).where(
            SourceFile.workspace_id == workspace_id,
            SourceFile.relative_path == relative_path,
        )
        if lock:
            statement = statement.with_for_update()
        return cast(SourceFile | None, await session.scalar(statement))


def _processing_job_record(job: ProcessingJob) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        job_id=job.job_id,
        workspace_id=job.workspace_id,
        input_path=job.input_path,
        total_count=job.total_count,
        completed_count=job.completed_count,
        failed_count=job.failed_count,
        status=job.status,
        current_stage=job.current_stage,
        error_info=list(job.error_info),
        stage_durations_ms=dict(job.stage_durations_ms),
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


async def _latest_embedding_states(
    session: AsyncSession,
    asset_ids: list[str],
) -> dict[str, list[AssetEmbeddingState]]:
    if not asset_ids:
        return {}
    rows = await session.scalars(
        select(EmbeddingRecord)
        .where(EmbeddingRecord.asset_id.in_(asset_ids))
        .order_by(
            EmbeddingRecord.asset_id,
            EmbeddingRecord.embedding_type,
            EmbeddingRecord.created_at.desc(),
            EmbeddingRecord.embedding_id.desc(),
        )
    )
    selected: dict[tuple[str, str], AssetEmbeddingState] = {}
    for record in rows:
        key = (record.asset_id, record.embedding_type)
        if key in selected:
            continue
        selected[key] = AssetEmbeddingState(
            embedding_type=record.embedding_type,
            status=record.status,
            model_name=record.model_name,
        )
    grouped: dict[str, list[AssetEmbeddingState]] = {}
    for (asset_id, _), state in selected.items():
        grouped.setdefault(asset_id, []).append(state)
    return grouped


def _asset_view_record(
    asset: Asset,
    source: SourceFile,
    *,
    embeddings: list[AssetEmbeddingState],
) -> AssetViewRecord:
    return AssetViewRecord(
        asset_id=asset.asset_id,
        workspace_id=asset.workspace_id,
        project_id=asset.project_id,
        source_file_id=asset.source_file_id,
        asset_type=AssetType(asset.asset_type),
        file_name=asset.file_name,
        file_type=asset.file_type,
        index_role=AssetIndexRole(asset.index_role),
        parent_asset_id=asset.parent_asset_id,
        child_order=asset.child_order,
        asset_name=asset.asset_name,
        asset_description=asset.asset_description,
        asset_features=dict(asset.asset_features),
        file_tree_context=list(asset.file_tree_context),
        source_contexts=list(asset.source_contexts),
        file_info=dict(asset.file_info),
        source_locator=dict(asset.source_locator),
        raw_content=asset.raw_content,
        source_storage_uri=source.storage_uri,
        derived_file_uri=asset.derived_file_uri,
        preview_uri=asset.preview_uri,
        processing_status=asset.processing_status,
        feature_revision=asset.feature_revision,
        embedding_revision=asset.embedding_revision,
        error_message=asset.error_message,
        source_file=AssetSourceRecord(
            source_file_id=source.source_file_id,
            original_file_name=source.original_file_name,
            relative_path=source.relative_path,
            file_type=source.file_type,
            mime_type=source.mime_type,
            file_size_bytes=source.file_size_bytes,
            processing_status=source.processing_status,
            error_message=source.error_message,
        ),
        embeddings=embeddings,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
    )


def _asset_values(values: AssetCreate) -> dict[str, object]:
    data = values.model_dump(mode="json")
    data["asset_type"] = values.asset_type.value
    data["index_role"] = values.index_role.value
    data["processing_status"] = values.processing_status.value
    if values.index_role == AssetIndexRole.PARENT:
        data["processing_status"] = ProcessingStatus.COMPLETED.value
    data["source_contexts"] = [context.model_dump() for context in values.source_contexts]
    return data


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _update_asset(current: Asset, values: AssetCreate, *, content_changed: bool) -> None:
    current.workspace_id = values.workspace_id
    current.asset_type = values.asset_type.value
    current.file_name = values.file_name
    current.file_type = values.file_type
    current.index_role = values.index_role.value
    current.child_order = values.child_order
    if values.index_role != AssetIndexRole.CHILD:
        current.parent_asset_id = None
    current.content_hash = values.content_hash
    current.generation = values.generation
    current.file_tree_context = values.file_tree_context
    current.source_contexts = [context.model_dump() for context in values.source_contexts]
    current.file_info = values.file_info
    current.source_locator = values.source_locator
    current.raw_content = values.raw_content
    current.derived_file_uri = values.derived_file_uri
    current.preview_uri = values.preview_uri
    current.processing_status = (
        ProcessingStatus.COMPLETED.value
        if values.index_role == AssetIndexRole.PARENT
        else values.processing_status.value
    )
    current.error_message = None
    if content_changed:
        if current.asset_name_source != AssetNameSource.USER.value:
            current.asset_name = None
            current.asset_name_source = None
        current.asset_description = None
        current.asset_features = {}
        current.feature_revision += 1
        current.embedding_revision += 1


def _validate_asset_hierarchy(
    assets: Sequence[AssetCreate],
    *,
    require_batch_parent: bool,
) -> None:
    """Reject invalid role/reference shapes before any row is changed."""
    by_key: dict[str, AssetCreate] = {}
    for asset in assets:
        if asset.asset_key in by_key:
            raise ValueError(f"duplicate asset_key in hierarchy batch: {asset.asset_key}")
        by_key[asset.asset_key] = asset
        if asset.index_role == AssetIndexRole.CHILD and not asset.parent_asset_key:
            raise ValueError("child assets require a parent_asset_key")

    if not require_batch_parent:
        return
    for asset in assets:
        if asset.index_role != AssetIndexRole.CHILD:
            continue
        parent = by_key.get(asset.parent_asset_key or "")
        if parent is None:
            raise ValueError("child asset parent does not exist in this batch")
        if parent.index_role != AssetIndexRole.PARENT:
            raise ValueError("child asset parent must have index_role=parent")


def _resolve_asset_parent(
    *,
    asset: Asset,
    values: AssetCreate,
    source_file_id: str,
    possible_parents: dict[str, Asset],
) -> None:
    """Link a child to a persisted parent in the same source file."""
    if values.index_role != AssetIndexRole.CHILD:
        asset.parent_asset_id = None
        return
    parent_key = values.parent_asset_key
    if parent_key is None:
        raise ValueError("child assets require a parent_asset_key")
    parent = possible_parents.get(parent_key)
    if parent is None:
        raise ValueError(f"child asset parent does not exist: {parent_key}")
    if parent.source_file_id != source_file_id:
        raise ValueError("child asset parent must belong to the same source file")
    if parent.index_role != AssetIndexRole.PARENT.value:
        raise ValueError("child asset parent must have index_role=parent")
    asset.parent_asset_id = parent.asset_id


# ===========================================
#      Embedding persistence
# ===========================================


@dataclass(slots=True, frozen=True)
class EmbeddingAsset:
    """The Asset fields needed to construct one model Embedding input."""

    asset_id: str
    workspace_id: str
    project_id: str
    source_file_id: str
    asset_type: str
    file_type: str
    content_hash: str
    embedding_revision: int
    created_at: datetime
    raw_content: str | None
    asset_description: str | None
    asset_features: dict[str, Any]
    derived_file_uri: str | None
    source_storage_uri: str
    source_mime_type: str
    file_name: str = ""
    source_relative_path: str = ""
    file_tree_context: list[str] = field(default_factory=list)
    source_contexts: list[dict[str, Any]] = field(default_factory=list)
    file_info: dict[str, Any] = field(default_factory=dict)
    source_locator: dict[str, Any] = field(default_factory=dict)
    index_role: str = AssetIndexRole.STANDALONE.value


@dataclass(slots=True, frozen=True)
class PreparedEmbedding:
    """A durable record reserved before calling the model or Milvus."""

    embedding_id: str
    milvus_primary_key: str
    already_indexed: bool


class EmbeddingRepository:
    """Own PostgreSQL metadata and state transitions for vector persistence."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._new_embedding_id = id_factory("emb")

    async def list_assets(
        self,
        *,
        workspace_id: str,
        asset_ids: Sequence[str] | None = None,
    ) -> list[EmbeddingAsset]:
        statement = (
            select(Asset, SourceFile)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                Asset.workspace_id == workspace_id,
                Asset.generation == SourceFile.processing_generation,
                Asset.index_role != AssetIndexRole.PARENT.value,
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
            )
            .order_by(Asset.created_at, Asset.asset_id)
        )
        if asset_ids:
            statement = statement.where(Asset.asset_id.in_(asset_ids))

        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()
        return [
            EmbeddingAsset(
                asset_id=asset.asset_id,
                workspace_id=asset.workspace_id,
                project_id=asset.project_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                file_type=asset.file_type,
                index_role=asset.index_role,
                content_hash=asset.content_hash,
                embedding_revision=asset.embedding_revision,
                created_at=asset.created_at,
                raw_content=asset.raw_content,
                asset_description=asset.asset_description,
                asset_features=dict(asset.asset_features),
                derived_file_uri=asset.derived_file_uri,
                source_storage_uri=source.storage_uri,
                source_mime_type=source.mime_type,
                file_name=asset.file_name,
                source_relative_path=source.relative_path,
                file_tree_context=list(asset.file_tree_context),
                source_contexts=list(asset.source_contexts),
                file_info=dict(asset.file_info),
                source_locator=dict(asset.source_locator),
            )
            for asset, source in rows
        ]

    async def prepare(
        self,
        *,
        asset: EmbeddingAsset,
        embedding_type: str,
        model_name: str,
        dimension: int,
        source_content_hash: str,
        source_mode: str,
        milvus_collection: str,
        force: bool,
    ) -> PreparedEmbedding:
        """Reserve one stable primary key or return an existing indexed record."""
        if asset.index_role == AssetIndexRole.PARENT.value:
            raise ValueError("parent assets must not receive embeddings")
        async with self._database.session() as session, session.begin():
            statement = (
                select(EmbeddingRecord)
                .where(
                    EmbeddingRecord.asset_id == asset.asset_id,
                    EmbeddingRecord.embedding_type == embedding_type,
                    EmbeddingRecord.model_name == model_name,
                    EmbeddingRecord.dimension == dimension,
                    EmbeddingRecord.source_content_hash == source_content_hash,
                )
                .with_for_update()
            )
            record = await session.scalar(statement)
            if record is not None:
                if record.status == EmbeddingStatus.INDEXED.value and not force:
                    return PreparedEmbedding(
                        embedding_id=record.embedding_id,
                        milvus_primary_key=record.milvus_primary_key,
                        already_indexed=True,
                    )
                record.status = EmbeddingStatus.PROCESSING.value
                record.latency_ms = None
                record.usage = {}
                return PreparedEmbedding(
                    embedding_id=record.embedding_id,
                    milvus_primary_key=record.milvus_primary_key,
                    already_indexed=False,
                )

            embedding_id = self._new_embedding_id()
            record = EmbeddingRecord(
                embedding_id=embedding_id,
                workspace_id=asset.workspace_id,
                project_id=asset.project_id,
                asset_id=asset.asset_id,
                embedding_type=embedding_type,
                model_name=model_name,
                dimension=dimension,
                source_content_hash=source_content_hash,
                embedding_source_mode=source_mode,
                milvus_collection=milvus_collection,
                milvus_primary_key=embedding_id,
                status=EmbeddingStatus.PROCESSING.value,
            )
            session.add(record)
            return PreparedEmbedding(
                embedding_id=embedding_id,
                milvus_primary_key=embedding_id,
                already_indexed=False,
            )

    async def mark_indexed(
        self,
        *,
        embedding_id: str,
        latency_ms: int,
        usage: dict[str, Any],
    ) -> None:
        async with self._database.session() as session, session.begin():
            record = await session.get(EmbeddingRecord, embedding_id, with_for_update=True)
            if record is None:
                raise ValueError(f"embedding record does not exist: {embedding_id}")
            record.status = EmbeddingStatus.INDEXED.value
            record.latency_ms = latency_ms
            record.usage = usage

    async def mark_failed(self, *, embedding_id: str) -> None:
        async with self._database.session() as session, session.begin():
            record = await session.get(EmbeddingRecord, embedding_id, with_for_update=True)
            if record is not None:
                record.status = EmbeddingStatus.FAILED.value

    async def list_indexed_embeddings(
        self,
        *,
        workspace_id: str,
        embedding_type: str,
        model_name: str,
        dimension: int,
        milvus_collection: str,
        asset_ids: Sequence[str] | None = None,
    ) -> list[IndexedEmbeddingAsset]:
        """Return the newest indexed vector metadata for each eligible Asset."""
        statement = (
            select(EmbeddingRecord, Asset, SourceFile.relative_path)
            .join(Asset, Asset.asset_id == EmbeddingRecord.asset_id)
            .join(SourceFile, SourceFile.source_file_id == Asset.source_file_id)
            .where(
                EmbeddingRecord.workspace_id == workspace_id,
                EmbeddingRecord.embedding_type == embedding_type,
                EmbeddingRecord.model_name == model_name,
                EmbeddingRecord.dimension == dimension,
                EmbeddingRecord.milvus_collection == milvus_collection,
                EmbeddingRecord.status == EmbeddingStatus.INDEXED.value,
                Asset.generation == SourceFile.processing_generation,
                Asset.index_role != AssetIndexRole.PARENT.value,
                Asset.processing_status != ProcessingStatus.SKIPPED.value,
            )
            .order_by(EmbeddingRecord.created_at.desc(), EmbeddingRecord.embedding_id.desc())
        )
        if asset_ids is not None:
            if not asset_ids:
                return []
            statement = statement.where(EmbeddingRecord.asset_id.in_(asset_ids))
        async with self._database.session() as session:
            rows = (await session.execute(statement)).all()

        selected: dict[str, IndexedEmbeddingAsset] = {}
        for record, asset, source_relative_path in rows:
            if asset.asset_id in selected:
                continue
            if not embedding_channel_is_eligible(
                embedding_type=embedding_type,
                asset_features=asset.asset_features,
                asset_description=asset.asset_description,
            ):
                continue
            selected[asset.asset_id] = IndexedEmbeddingAsset(
                embedding_id=record.embedding_id,
                asset_id=asset.asset_id,
                source_file_id=asset.source_file_id,
                asset_type=asset.asset_type,
                asset_name=asset.asset_name,
                asset_description=asset.asset_description,
                asset_features=dict(asset.asset_features),
                file_tree_context=list(asset.file_tree_context),
                source_relative_path=source_relative_path,
            )
        return list(selected.values())

def _collect_storage_uris(value: Any, collected: set[str]) -> None:
    """Collect only explicit storage URIs from nested persisted metadata."""
    if isinstance(value, str):
        if value.startswith(("s3://", "file://")):
            collected.add(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_storage_uris(item, collected)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_storage_uris(item, collected)
