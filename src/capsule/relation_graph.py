"""Build an Asset relationship graph from independent metadata and content paths."""

import hashlib
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from capsule.db.repositories import EmbeddingAsset
from capsule.schemas import AssetUnderstanding, SubjectFeatureItem

_GENERIC_DIRECTORIES = frozenset(
    {
        "asset",
        "assets",
        "data",
        "file",
        "files",
        "image",
        "images",
        "download",
        "downloads",
        "素材",
        "图片",
        "图像",
        "下载",
        "未分类",
    }
)


class AssetEntityRelationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    establishes_relation: bool
    relation: str
    description: str


class MetadataContentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_entity: str
    relation: Literal["same_entity", "contains_content", "related"]
    description: str
    entity_semantic: str = ""
    build_entity: bool = True


class MetadataContentResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[MetadataContentDecision] = Field(default_factory=list)


class AssetEntityRelationResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relations: list[AssetEntityRelationDecision] = Field(default_factory=list)


class MergedEntityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    semantic: str
    candidate_ids: list[str]
    build_entity: bool
    reason: str

    @field_validator("reason", mode="before")
    @classmethod
    def compact_reason(cls, value: object) -> str:
        text = str(value or "").strip()
        return text if len(text) <= 60 else f"{text[:59]}…"


class MergedEntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[MergedEntityDecision] = Field(default_factory=list)


class EntityStructureNode(BaseModel):
    """An Entity visible to one incremental structure-building round."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=200)
    semantic: str = Field(min_length=1, max_length=1000)


class EntityStructureEdge(BaseModel):
    """An already accepted Entity-to-Entity edge in the current structure."""

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str = Field(min_length=1, max_length=256)
    target_entity_id: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)


class CurrentEntityStructure(BaseModel):
    """The complete structure maintained by the backend between Agent rounds."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[EntityStructureNode] = Field(default_factory=list)
    edges: list[EntityStructureEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "CurrentEntityStructure":
        node_ids = [node.entity_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("current_graph contains duplicate entity_id values")
        known_ids = set(node_ids)
        for edge in self.edges:
            if edge.source_entity_id not in known_ids or edge.target_entity_id not in known_ids:
                raise ValueError("current_graph edge references an unknown entity_id")
            if edge.source_entity_id == edge.target_entity_id:
                raise ValueError("current_graph cannot contain a self edge")
        return self


class EntityStructureRequest(BaseModel):
    """Stable input envelope sent to the Entity structure Agent."""

    model_config = ConfigDict(extra="forbid")

    workspace_tree: str = Field(min_length=1)
    current_graph: CurrentEntityStructure
    incoming_entities: list[EntityStructureNode] = Field(min_length=1, max_length=15)

    @model_validator(mode="after")
    def validate_entity_ids(self) -> "EntityStructureRequest":
        incoming_ids = [entity.entity_id for entity in self.incoming_entities]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("incoming_entities contains duplicate entity_id values")
        return self


class MergeEntityOperation(BaseModel):
    """Merge Entity nodes that denote virtually the same real Entity."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["merge"]
    source_entity_ids: list[str] = Field(min_length=2)
    canonical_entity_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=200)
    semantic: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_source_ids(self) -> "MergeEntityOperation":
        if len(self.source_entity_ids) != len(set(self.source_entity_ids)):
            raise ValueError("merge source_entity_ids must be unique")
        if self.canonical_entity_id not in self.source_entity_ids:
            raise ValueError("canonical_entity_id must occur in source_entity_ids")
        return self


class SeparateEntityOperation(BaseModel):
    """Keep unrelated Entity nodes independent in this round."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["separate"]
    entity_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity_ids(self) -> "SeparateEntityOperation":
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("separate entity_ids must be unique")
        return self


class ReusedGroupParent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["reuse"]
    parent_entity_id: str = Field(min_length=1, max_length=256)


class CreatedGroupParent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["create"]
    temporary_parent_id: str = Field(pattern=r"^virtual:[^\s/]+$", max_length=256)
    name: str = Field(min_length=1, max_length=200)
    semantic: str = Field(min_length=1, max_length=1000)


GroupParent = Annotated[
    ReusedGroupParent | CreatedGroupParent,
    Field(discriminator="mode"),
]


class GroupChildRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_entity_id: str = Field(min_length=1, max_length=256)
    relation: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)


class GroupEntityOperation(BaseModel):
    """Attach related-but-distinct Entities to a reused or newly created parent."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["group"]
    parent: GroupParent
    children: list[GroupChildRelation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_children(self) -> "GroupEntityOperation":
        child_ids = [child.child_entity_id for child in self.children]
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("group child_entity_id values must be unique")
        if (
            isinstance(self.parent, ReusedGroupParent)
            and self.parent.parent_entity_id in child_ids
        ):
            raise ValueError("a reused parent cannot also be its own child")
        return self


EntityStructureOperation = Annotated[
    MergeEntityOperation | SeparateEntityOperation | GroupEntityOperation,
    Field(discriminator="type"),
]


class EntityStructureOperationResolution(BaseModel):
    """Incremental operations only; the backend remains owner of the full graph."""

    model_config = ConfigDict(extra="forbid")

    operations: list[EntityStructureOperation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_operation_ids(self) -> "EntityStructureOperationResolution":
        temporary_parent_ids: list[str] = []
        for operation in self.operations:
            if isinstance(operation, GroupEntityOperation) and isinstance(
                operation.parent, CreatedGroupParent
            ):
                temporary_parent_ids.append(operation.parent.temporary_parent_id)
        if len(temporary_parent_ids) != len(set(temporary_parent_ids)):
            raise ValueError("temporary_parent_id values must be unique")
        return self


def _entity_key(subject: str) -> str:
    normalized = unicodedata.normalize("NFKC", subject).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _metadata_names(asset: EmbeddingAsset) -> list[str]:
    """Extract collection-level names without inspecting Asset content."""

    if not asset.source_relative_path:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for part in PurePosixPath(asset.source_relative_path).parts[:-1]:
        name = part.strip()
        key = _entity_key(name)
        if not key or key in _GENERIC_DIRECTORIES or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _primary_content_subject(
    understanding: AssetUnderstanding,
) -> SubjectFeatureItem | None:
    return max(
        understanding.features.subject_content.items,
        key=lambda item: item.salience,
        default=None,
    )


def _group_relation(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        relation = value.get("relation")
        return relation if isinstance(relation, str) else None
    return None


def _entity_semantic(value: object, fallback: str) -> str:
    if isinstance(value, Mapping):
        semantic = value.get("entity_semantic")
        if isinstance(semantic, str) and semantic.strip():
            return semantic.strip()
    return fallback


def _should_build_entity(value: object) -> bool:
    if isinstance(value, Mapping):
        build_entity = value.get("build_entity")
        if isinstance(build_entity, bool):
            return build_entity
    return True


def _main_entities(
    *,
    asset: EmbeddingAsset,
    understanding: AssetUnderstanding,
    reusable_metadata_keys: set[str],
    metadata_content_relations: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Merge exact metadata/content matches and preserve different entity layers."""

    merged: dict[str, dict[str, Any]] = {}
    for name in _metadata_names(asset):
        key = _entity_key(name)
        if key not in reusable_metadata_keys:
            continue
        if not _should_build_entity(metadata_content_relations.get(key)):
            continue
        merged[key] = {
            "subject": name,
            "description": f"元数据中可跨素材复用的实体：{name}",
            "salience": 1.0,
            "origins": ["metadata"],
        }

    content = _primary_content_subject(understanding)
    if content is not None:
        key = _entity_key(content.subject)
        matching_metadata_key = next(
            (
                metadata_key
                for metadata_key in merged
                if metadata_key == key
                or _group_relation(metadata_content_relations.get(metadata_key)) == "same_entity"
            ),
            None,
        )
        current = merged.get(matching_metadata_key) if matching_metadata_key else None
        if current is None:
            merged[key] = {
                "subject": content.subject,
                "description": content.description,
                "salience": content.salience,
                "origins": ["content"],
            }
        else:
            current["description"] = content.description
            current["salience"] = max(float(current["salience"]), content.salience)
            current["origins"].append("content")
    return list(merged.values())


def build_relation_graph(
    assets: list[EmbeddingAsset],
    understandings: dict[str, AssetUnderstanding],
    *,
    metadata_content_relations: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Extract metadata/content entities separately, then merge and build edges."""

    metadata_counts = Counter(
        _entity_key(name) for asset in assets for name in _metadata_names(asset)
    )
    reusable_metadata_keys = {key for key, count in metadata_counts.items() if count >= 2}
    resolved_relations = metadata_content_relations or {}
    asset_rows: list[dict[str, Any]] = []
    grouped_members: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for asset in assets:
        understanding = understandings[asset.asset_id]
        main_entities = _main_entities(
            asset=asset,
            understanding=understanding,
            reusable_metadata_keys=reusable_metadata_keys,
            metadata_content_relations=resolved_relations,
        )
        primary = _primary_content_subject(understanding)
        row: dict[str, Any] = {
            "asset_id": asset.asset_id,
            "source_path": asset.source_relative_path,
            "asset_name": understanding.asset_name,
            "asset_description": understanding.asset_description,
            "primary_subject": (
                {
                    "subject": primary.subject,
                    "description": primary.description,
                    "status": primary.status.value,
                }
                if primary is not None
                else None
            ),
            "main_entities": main_entities,
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
        asset_rows.append(row)
        for entity in main_entities:
            grouped_members[_entity_key(entity["subject"])].append((row, entity))

    virtual_entity_keys = {
        entity_key
        for entity_key, members in grouped_members.items()
        if len({row["asset_id"] for row, _ in members}) >= 2
    }
    for row in asset_rows:
        row["main_entities"] = [
            entity
            for entity in row["main_entities"]
            if _entity_key(entity["subject"]) in virtual_entity_keys
        ]

    entities: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    entity_id_by_key: dict[str, str] = {}
    for entity_key, members in sorted(grouped_members.items()):
        if entity_key not in virtual_entity_keys:
            continue
        subject = members[0][1]["subject"]
        entity_id = f"entity_{hashlib.sha256(entity_key.encode()).hexdigest()[:12]}"
        entity_id_by_key[entity_key] = entity_id
        origins = sorted({origin for _, entity in members for origin in entity["origins"]})
        descriptions = list(dict.fromkeys(entity["description"] for _, entity in members))
        semantic = _entity_semantic(
            resolved_relations.get(entity_key),
            descriptions[0] if descriptions else subject,
        )
        entities.append(
            {
                "entity_id": entity_id,
                "name": subject,
                "semantic": semantic,
                "origins": origins,
                "asset_ids": [row["asset_id"] for row, _ in members],
                "descriptions": descriptions,
            }
        )
        for row, entity in members:
            relation = (
                "HAS_MERGED_ENTITY"
                if len(entity["origins"]) > 1
                else (
                    "HAS_METADATA_ENTITY"
                    if entity["origins"] == ["metadata"]
                    else "HAS_CONTENT_ENTITY"
                )
            )
            edges.append(
                {
                    "source": row["asset_id"],
                    "target": entity_id,
                    "relation": relation,
                    "description": "",
                    "content_subject": (
                        row["primary_subject"]["subject"]
                        if row["primary_subject"] is not None
                        else ""
                    ),
                }
            )
        shared_relation = "SHARES_METADATA_ENTITY" if origins == ["metadata"] else "SAME_ENTITY"
        for left_index, (left, _) in enumerate(members):
            for right, _ in members[left_index + 1 :]:
                edges.append(
                    {
                        "source": left["asset_id"],
                        "target": right["asset_id"],
                        "relation": shared_relation,
                        "description": f"两项素材共同关联实体{subject}",
                    }
                )

    emitted_entity_relations: set[tuple[str, str, str]] = set()
    for row in asset_rows:
        metadata_entities = [
            entity for entity in row["main_entities"] if entity["origins"] == ["metadata"]
        ]
        content_entities = [
            entity for entity in row["main_entities"] if entity["origins"] == ["content"]
        ]
        for metadata_entity in metadata_entities:
            metadata_key = _entity_key(metadata_entity["subject"])
            resolved_relation = _group_relation(resolved_relations.get(metadata_key))
            if (
                resolved_relation not in {"contains_content", "related"}
                or metadata_key not in virtual_entity_keys
            ):
                continue
            for content_entity in content_entities:
                content_key = _entity_key(content_entity["subject"])
                if content_key not in virtual_entity_keys:
                    continue
                graph_relation = (
                    "CONTAINS" if resolved_relation == "contains_content" else "RELATED_TO"
                )
                edge_key = (metadata_key, content_key, graph_relation)
                if edge_key in emitted_entity_relations:
                    continue
                emitted_entity_relations.add(edge_key)
                edges.append(
                    {
                        "source": entity_id_by_key[metadata_key],
                        "target": entity_id_by_key[content_key],
                        "relation": graph_relation,
                        "description": (
                            f"{metadata_entity['subject']}包含内容实体{content_entity['subject']}"
                            if graph_relation == "CONTAINS"
                            else (
                                f"{metadata_entity['subject']}与内容实体"
                                f"{content_entity['subject']}相关"
                            )
                        ),
                    }
                )

    return {
        "asset_count": len(asset_rows),
        "entity_count": len(entities),
        "edge_count": len(edges),
        "assets": asset_rows,
        "entities": entities,
        "edges": edges,
    }


def build_merged_candidate_graph(
    assets: list[EmbeddingAsset],
    understandings: dict[str, AssetUnderstanding],
    *,
    candidates: Sequence[Mapping[str, Any]],
    resolution: MergedEntityResolution,
) -> dict[str, Any]:
    """Build provisional Asset→Entity edges from merged candidate memberships."""

    graph = build_relation_graph(assets, understandings)
    graph["entities"] = []
    graph["edges"] = []
    assets_by_id = {row["asset_id"]: row for row in graph["assets"]}
    for row in graph["assets"]:
        row["main_entities"] = []
    candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}

    for decision in resolution.entities:
        if not decision.build_entity:
            continue
        selected = [
            candidates_by_id[candidate_id]
            for candidate_id in dict.fromkeys(decision.candidate_ids)
            if candidate_id in candidates_by_id
        ]
        member_ids = list(
            dict.fromkeys(
                str(asset_id)
                for candidate in selected
                for asset_id in candidate.get("asset_ids", [])
                if str(asset_id) in assets_by_id
            )
        )
        if len(member_ids) < 2:
            continue
        entity_key = _entity_key(decision.name)
        identity = "|".join(sorted(decision.candidate_ids)) or entity_key
        entity_id = f"entity_{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        origins = sorted({str(candidate.get("origin", "candidate")) for candidate in selected})
        descriptions = list(
            dict.fromkeys(
                str(candidate.get("semantic", ""))
                for candidate in selected
                if str(candidate.get("semantic", "")).strip()
            )
        )
        graph["entities"].append(
            {
                "entity_id": entity_id,
                "name": decision.name,
                "semantic": decision.semantic,
                "origins": origins,
                "asset_ids": member_ids,
                "descriptions": descriptions,
                "candidate_ids": decision.candidate_ids,
                "merge_reason": decision.reason,
            }
        )
        for asset_id in member_ids:
            row = assets_by_id[asset_id]
            is_subject_cluster_member = any(
                candidate.get("origin") == "subject_cluster"
                and asset_id in candidate.get("asset_ids", [])
                for candidate in selected
            )
            row["main_entities"].append(
                {
                    "subject": decision.name,
                    "description": decision.semantic,
                    "salience": 1.0,
                    "origins": origins,
                }
            )
            graph["edges"].append(
                {
                    "source": asset_id,
                    "target": entity_id,
                    "relation": (
                        "CLUSTER_MEMBER" if is_subject_cluster_member else "CANDIDATE_MEMBER"
                    ),
                    "description": ("内容高度相似" if is_subject_cluster_member else ""),
                    "content_subject": (
                        row["primary_subject"]["subject"]
                        if row["primary_subject"] is not None
                        else ""
                    ),
                }
            )

    graph["entity_count"] = len(graph["entities"])
    graph["edge_count"] = len(graph["edges"])
    return graph


def apply_asset_entity_relations(
    graph: dict[str, Any],
    resolution: AssetEntityRelationResolution,
) -> None:
    """Apply independently generated two-node relationship judgments."""

    decisions = {(item.source_id, item.target_id): item for item in resolution.relations}
    entity_ids = {entity["entity_id"] for entity in graph["entities"]}
    retained_edges: list[dict[str, Any]] = []
    rejected_relations: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in _deduplicate_asset_entity_edges(graph["edges"]):
        if edge["source"] in entity_ids and edge["target"] in entity_ids:
            retained_edges.append(edge)
            continue
        if edge["target"] not in entity_ids:
            continue
        decision = decisions.get((edge["source"], edge["target"]))
        if decision is None:
            if edge["relation"] == "CLUSTER_MEMBER":
                retained_edges.append(edge)
            continue
        if not decision.establishes_relation:
            rejected_relations[(decision.source_id, decision.target_id)] = (
                decision.model_dump(mode="json")
            )
            continue
        edge["relation"] = decision.relation
        edge["description"] = decision.description
        retained_edges.append(edge)

    member_ids_by_entity: dict[str, set[str]] = defaultdict(set)
    for edge in retained_edges:
        if edge["target"] in entity_ids and edge["source"] not in entity_ids:
            member_ids_by_entity[edge["target"]].add(edge["source"])
    retained_entity_ids = {
        entity_id for entity_id, member_ids in member_ids_by_entity.items() if len(member_ids) >= 2
    }
    graph["entities"] = [
        entity for entity in graph["entities"] if entity["entity_id"] in retained_entity_ids
    ]
    for entity in graph["entities"]:
        entity["asset_ids"] = sorted(member_ids_by_entity[entity["entity_id"]])
    graph["edges"] = [
        edge
        for edge in retained_edges
        if (edge["source"] not in entity_ids or edge["source"] in retained_entity_ids)
        and (edge["target"] not in entity_ids or edge["target"] in retained_entity_ids)
    ]
    graph["entity_count"] = len(graph["entities"])
    graph["edge_count"] = len(graph["edges"])
    graph["rejected_relations"] = list(rejected_relations.values())


def apply_entity_structure_operations(
    graph: dict[str, Any],
    resolution: EntityStructureOperationResolution,
) -> list[str]:
    """Apply one validated Agent round and return newly created virtual Entity IDs."""

    graph.setdefault("entity_edges", [])
    created_ids: list[str] = []
    for operation in resolution.operations:
        if isinstance(operation, MergeEntityOperation):
            _apply_entity_merge(graph, operation)
        elif isinstance(operation, GroupEntityOperation):
            if isinstance(operation.parent, CreatedGroupParent) and len(operation.children) < 2:
                continue
            existing_ids = {str(node["entity_id"]) for node in graph["entities"]}
            parent_id = _resolve_group_parent(graph, operation.parent)
            if (
                isinstance(operation.parent, CreatedGroupParent)
                and parent_id not in existing_ids
            ):
                created_ids.append(parent_id)
            for child in operation.children:
                if child.child_entity_id == parent_id:
                    continue
                _upsert_entity_edge(
                    graph,
                    source_entity_id=child.child_entity_id,
                    target_entity_id=parent_id,
                    relation=child.relation,
                    description=child.description,
                )
        elif isinstance(operation, SeparateEntityOperation):
            continue

    graph["entity_count"] = len(graph["entities"])
    graph["entity_edge_count"] = len(graph["entity_edges"])
    graph["edge_count"] = len(graph["edges"]) + len(graph["entity_edges"])
    return list(dict.fromkeys(created_ids))


def _apply_entity_merge(graph: dict[str, Any], operation: MergeEntityOperation) -> None:
    nodes_by_id = {node["entity_id"]: node for node in graph["entities"]}
    selected = [
        nodes_by_id[entity_id]
        for entity_id in operation.source_entity_ids
        if entity_id in nodes_by_id
    ]
    if len(selected) < 2:
        return
    canonical = nodes_by_id.get(operation.canonical_entity_id)
    if canonical is None:
        return

    canonical["name"] = operation.name
    canonical["semantic"] = operation.semantic
    for field in ("asset_ids", "origins", "descriptions", "candidate_ids"):
        canonical[field] = list(
            dict.fromkeys(
                str(value)
                for node in selected
                for value in node.get(field, [])
            )
        )
    canonical["embedding_vector"] = next(
        (
            list(node.get("embedding_vector", []))
            for node in selected
            if node.get("embedding_vector")
        ),
        [],
    )
    canonical["embedding_model"] = next(
        (str(node.get("embedding_model", "")) for node in selected if node.get("embedding_model")),
        "",
    )

    replaced_ids = set(operation.source_entity_ids) - {operation.canonical_entity_id}
    graph["entities"] = [
        node for node in graph["entities"] if node["entity_id"] not in replaced_ids
    ]
    for edge in graph["edges"]:
        if edge.get("target") in replaced_ids:
            edge["target"] = operation.canonical_entity_id
    graph["edges"] = _deduplicate_asset_entity_edges(graph["edges"])

    rewritten_entity_edges: list[dict[str, Any]] = []
    for edge in graph["entity_edges"]:
        source_id = edge["source_entity_id"]
        target_id = edge["target_entity_id"]
        if source_id in replaced_ids:
            source_id = operation.canonical_entity_id
        if target_id in replaced_ids:
            target_id = operation.canonical_entity_id
        if source_id == target_id:
            continue
        rewritten_entity_edges.append(
            {
                **edge,
                "source_entity_id": source_id,
                "target_entity_id": target_id,
            }
        )
    graph["entity_edges"] = _deduplicate_entity_edges(rewritten_entity_edges)


def _resolve_group_parent(
    graph: dict[str, Any],
    parent: GroupParent,
) -> str:
    if isinstance(parent, ReusedGroupParent):
        return parent.parent_entity_id

    identity = f"{_entity_key(parent.name)}\0{parent.semantic.strip()}"
    parent_id = f"entity_virtual_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    existing = next(
        (node for node in graph["entities"] if node["entity_id"] == parent_id),
        None,
    )
    if existing is None:
        graph["entities"].append(
            {
                "entity_id": parent_id,
                "name": parent.name,
                "semantic": parent.semantic,
                "origins": ["agent_structure"],
                "asset_ids": [],
                "descriptions": [parent.semantic],
                "candidate_ids": [],
                "merge_reason": "",
                "embedding_vector": [],
                "embedding_model": "",
            }
        )
    return parent_id


def _upsert_entity_edge(
    graph: dict[str, Any],
    *,
    source_entity_id: str,
    target_entity_id: str,
    relation: str,
    description: str,
) -> None:
    for edge in graph["entity_edges"]:
        if (
            edge["source_entity_id"] == source_entity_id
            and edge["target_entity_id"] == target_entity_id
        ):
            edge["relation"] = relation
            edge["description"] = description
            return
    graph["entity_edges"].append(
        {
            "source_entity_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "relation": relation,
            "description": description,
        }
    )


def _deduplicate_asset_entity_edges(
    edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (str(edge["source"]), str(edge["target"]))
        current = selected.get(key)
        if current is None or edge.get("relation") == "CLUSTER_MEMBER":
            selected[key] = edge
    return list(selected.values())


def _deduplicate_entity_edges(
    edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        selected[
            (str(edge["source_entity_id"]), str(edge["target_entity_id"]))
        ] = edge
    return list(selected.values())
