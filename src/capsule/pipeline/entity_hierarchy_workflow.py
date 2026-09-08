"""LangGraph workflow for incrementally maintaining the Entity hierarchy.

The graph deliberately persists only coordination state.  Entity trees and an
Agent's structured decision are held in the process-local operation cache for
one node transition, then re-hydrated from the repository after a restart.
PostgreSQL remains the source of truth for both hierarchy data and idempotent
batch commits.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from capsule.relation_graph import (
    CreatedGroupParent,
    EntityStructureEdge,
    EntityStructureNode,
    EntityStructureOperationResolution,
    GroupEntityOperation,
    MergeEntityOperation,
    ReusedGroupParent,
    SeparateEntityOperation,
)

HIERARCHY_BATCH_SIZE = 10
TOP_K_ENTITY_NODES = 3


class EntityHierarchyWorkflowError(RuntimeError):
    """Raised when the Agent cannot produce one valid decision after repair."""


class AgentToolScopeError(ValueError):
    """Raised when an Agent tries to read outside its retrieved candidate scope."""


class HierarchyOperationValidationError(ValueError):
    """A structured Agent decision violates the hierarchy write contract."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class SimilarEntityNodeHit(BaseModel):
    """One vector-search hit, resolved from a node to its complete root tree."""

    model_config = ConfigDict(extra="forbid")

    incoming_entity_id: str = Field(min_length=1, max_length=256)
    entity_id: str = Field(min_length=1, max_length=256)
    root_entity_id: str = Field(min_length=1, max_length=256)
    score: float


class CandidateEntityTreeSelection(BaseModel):
    """The only root trees the current Agent invocation may inspect."""

    model_config = ConfigDict(extra="forbid")

    hits: list[SimilarEntityNodeHit] = Field(default_factory=list)
    root_entity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_roots(self) -> CandidateEntityTreeSelection:
        if len(self.root_entity_ids) != len(set(self.root_entity_ids)):
            raise ValueError("root_entity_ids must be unique")
        return self


class EntityHierarchyTree(BaseModel):
    """A complete hierarchy tree with no raw Asset content attached."""

    model_config = ConfigDict(extra="forbid")

    root_entity_id: str = Field(min_length=1, max_length=256)
    nodes: list[EntityStructureNode] = Field(default_factory=list)
    edges: list[EntityStructureEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tree(self) -> EntityHierarchyTree:
        node_ids = [node.entity_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("tree nodes contain duplicate entity_id values")
        known_ids = set(node_ids)
        if self.root_entity_id not in known_ids:
            raise ValueError("tree root_entity_id must occur in nodes")
        for edge in self.edges:
            if edge.source_entity_id not in known_ids or edge.target_entity_id not in known_ids:
                raise ValueError("tree edge references an entity outside the tree")
            if edge.source_entity_id == edge.target_entity_id:
                raise ValueError("tree cannot contain a self edge")
        return self


class HierarchyAgentContext(BaseModel):
    """Small, stable payload supplied to the Agent before it calls read tools."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    input_revision: str = Field(min_length=1, max_length=256)
    incoming_entities: list[EntityStructureNode] = Field(
        min_length=1,
        max_length=HIERARCHY_BATCH_SIZE,
    )

    @model_validator(mode="after")
    def validate_incoming_entities(self) -> HierarchyAgentContext:
        entity_ids = [entity.entity_id for entity in self.incoming_entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("incoming_entities must be unique")
        return self


class HierarchyCommitRequest(BaseModel):
    """Validated, idempotent unit of hierarchy persistence owned by the backend."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(min_length=1)
    input_revision: str = Field(min_length=1, max_length=256)
    job_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1, max_length=256)
    incoming_entity_ids: list[str] = Field(min_length=1, max_length=HIERARCHY_BATCH_SIZE)
    candidate_root_entity_ids: list[str] = Field(default_factory=list)
    resolution: EntityStructureOperationResolution
    created_parent_ids_by_temporary_id: dict[str, str] = Field(default_factory=dict)


class HierarchyCommitResult(BaseModel):
    """Only actually inserted parents are returned for a later queue round."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=256)
    created_parent_entity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_created_parents(self) -> HierarchyCommitResult:
        if len(self.created_parent_entity_ids) != len(set(self.created_parent_entity_ids)):
            raise ValueError("created_parent_entity_ids must be unique")
        return self


@runtime_checkable
class EntityHierarchyRepository(Protocol):
    """Storage and index boundary; implementations own SQL transactions and locks."""

    async def load_entity_nodes(
        self,
        *,
        workspace_id: str,
        entity_ids: Sequence[str],
    ) -> Sequence[EntityStructureNode]:
        """Load name and semantic text for the exact incoming Entity IDs."""

    async def search_similar_entity_nodes(
        self,
        *,
        workspace_id: str,
        incoming_entity_ids: Sequence[str],
        exclude_entity_ids: Sequence[str],
        per_entity_limit: int,
    ) -> Sequence[SimilarEntityNodeHit]:
        """Return at most ``per_entity_limit`` existing node hits per incoming Entity.

        Each hit must contain the matched node and the root Entity of its full
        hierarchy tree.  The repository must use the Entity semantic vector
        index, not raw Asset vectors.
        """

    async def load_complete_entity_trees(
        self,
        *,
        workspace_id: str,
        root_entity_ids: Sequence[str],
    ) -> Sequence[EntityHierarchyTree]:
        """Load every node and hierarchy edge under each requested root only."""

    async def load_workspace_tree(self, *, workspace_id: str) -> str:
        """Return the workspace directory tree without raw Asset contents."""

    async def commit_hierarchy_batch(
        self,
        request: HierarchyCommitRequest,
    ) -> HierarchyCommitResult:
        """Transactionally persist one validated operation ID exactly once.

        The implementation must version-check ``input_revision``, use the
        operation ID as an idempotency key, create parents with the supplied
        stable IDs, and return only newly inserted parent IDs.
        """


@runtime_checkable
class EntityHierarchyAgent(Protocol):
    """An Agent may inspect bounded read tools but never receives a write tool."""

    async def decide(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
    ) -> EntityStructureOperationResolution:
        """Produce merge, separate, and group operations for one batch."""

    async def repair(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
        previous_resolution: EntityStructureOperationResolution | None,
        validation_errors: Sequence[str],
    ) -> EntityStructureOperationResolution:
        """Repair one invalid decision without requesting human review."""


@dataclass(frozen=True, slots=True)
class OperationScope:
    """Ephemeral write scope derived from the Agent's bounded read calls."""

    incoming_entities: tuple[EntityStructureNode, ...]
    candidate_trees: tuple[EntityHierarchyTree, ...]
    workspace_tree: str

    @property
    def incoming_entity_ids(self) -> frozenset[str]:
        return frozenset(entity.entity_id for entity in self.incoming_entities)

    @property
    def candidate_root_entity_ids(self) -> tuple[str, ...]:
        return tuple(tree.root_entity_id for tree in self.candidate_trees)

    @property
    def candidate_entity_ids(self) -> frozenset[str]:
        return frozenset(node.entity_id for tree in self.candidate_trees for node in tree.nodes)

    @property
    def hierarchy_edges(self) -> tuple[EntityStructureEdge, ...]:
        return tuple(edge for tree in self.candidate_trees for edge in tree.edges)


class BoundedEntityHierarchyTools:
    """Read-only Agent tools whose scope is cryptographically unnecessary but hard-bound.

    The Agent can query all ten entities in its assigned batch once, then can
    read exactly the deduplicated complete root trees returned by that query.
    It cannot search arbitrary node IDs, fetch a global graph, or write data.
    """

    def __init__(
        self,
        *,
        repository: EntityHierarchyRepository,
        workspace_id: str,
        incoming_entities: Sequence[EntityStructureNode],
        excluded_entity_ids: Sequence[str] = (),
    ) -> None:
        self._repository = repository
        self._workspace_id = workspace_id
        self._incoming_entities = tuple(incoming_entities)
        self._incoming_ids = tuple(entity.entity_id for entity in incoming_entities)
        self._excluded_entity_ids = frozenset([*self._incoming_ids, *excluded_entity_ids])
        self._selection: CandidateEntityTreeSelection | None = None
        self._trees: tuple[EntityHierarchyTree, ...] | None = None
        self._workspace_tree: str | None = None

    async def retrieve_related_entity_trees(
        self,
        entity_ids: Sequence[str],
    ) -> CandidateEntityTreeSelection:
        """Find Top3 existing nodes for every batch Entity and deduplicate roots."""

        requested = tuple(entity_ids)
        if requested != self._incoming_ids:
            raise AgentToolScopeError(
                "retrieve_related_entity_trees only accepts the exact current batch"
            )
        if self._selection is not None:
            return self._selection

        raw_hits = await self._repository.search_similar_entity_nodes(
            workspace_id=self._workspace_id,
            incoming_entity_ids=self._incoming_ids,
            exclude_entity_ids=tuple(sorted(self._excluded_entity_ids)),
            per_entity_limit=TOP_K_ENTITY_NODES,
        )
        hits = [
            hit
            if isinstance(hit, SimilarEntityNodeHit)
            else SimilarEntityNodeHit.model_validate(hit)
            for hit in raw_hits
        ]
        errors: list[str] = []
        hits_by_incoming: dict[str, list[SimilarEntityNodeHit]] = {
            entity_id: [] for entity_id in self._incoming_ids
        }
        for hit in hits:
            if hit.incoming_entity_id not in hits_by_incoming:
                errors.append(
                    "vector search returned a hit for unknown incoming Entity "
                    f"{hit.incoming_entity_id}"
                )
                continue
            if (
                hit.entity_id in self._excluded_entity_ids
                or hit.root_entity_id in self._excluded_entity_ids
            ):
                errors.append("vector search returned an unprocessed Entity as a candidate tree")
                continue
            hits_by_incoming[hit.incoming_entity_id].append(hit)
        for entity_id, entity_hits in hits_by_incoming.items():
            if len(entity_hits) > TOP_K_ENTITY_NODES:
                errors.append(
                    f"vector search returned more than Top{TOP_K_ENTITY_NODES} for {entity_id}"
                )
        if errors:
            raise AgentToolScopeError("; ".join(errors))

        root_ids: list[str] = []
        seen_roots: set[str] = set()
        for entity_id in self._incoming_ids:
            for hit in hits_by_incoming[entity_id]:
                if hit.root_entity_id not in seen_roots:
                    seen_roots.add(hit.root_entity_id)
                    root_ids.append(hit.root_entity_id)
        self._selection = CandidateEntityTreeSelection(hits=hits, root_entity_ids=root_ids)
        return self._selection

    async def read_complete_entity_trees(
        self,
        root_entity_ids: Sequence[str],
    ) -> tuple[EntityHierarchyTree, ...]:
        """Read all nodes and hierarchy edges for exactly the retrieved roots."""

        if self._selection is None:
            raise AgentToolScopeError(
                "retrieve_related_entity_trees must run before read_complete_entity_trees"
            )
        requested = tuple(root_entity_ids)
        expected = tuple(self._selection.root_entity_ids)
        if requested != expected:
            raise AgentToolScopeError(
                "read_complete_entity_trees only accepts the deduplicated candidate roots"
            )
        if self._trees is not None:
            return self._trees

        raw_trees = await self._repository.load_complete_entity_trees(
            workspace_id=self._workspace_id,
            root_entity_ids=expected,
        )
        trees = tuple(
            tree
            if isinstance(tree, EntityHierarchyTree)
            else EntityHierarchyTree.model_validate(tree)
            for tree in raw_trees
        )
        returned_roots = tuple(tree.root_entity_id for tree in trees)
        if len(returned_roots) != len(set(returned_roots)) or set(returned_roots) != set(expected):
            raise AgentToolScopeError(
                "repository must return one complete tree for every candidate root"
            )
        self._trees = trees
        return self._trees

    async def read_workspace_tree(self) -> str:
        """Read the directory-only workspace context for this batch."""

        if self._workspace_tree is None:
            tree = await self._repository.load_workspace_tree(workspace_id=self._workspace_id)
            if not isinstance(tree, str) or not tree.strip():
                raise AgentToolScopeError("workspace tree must be a non-empty string")
            self._workspace_tree = tree
        return self._workspace_tree

    def operation_scope(self) -> OperationScope:
        """Expose the write scope only after the Agent has consumed all read context."""

        if self._selection is None:
            raise AgentToolScopeError("Agent did not retrieve related Entity trees")
        if self._trees is None:
            raise AgentToolScopeError("Agent did not read the complete candidate Entity trees")
        if self._workspace_tree is None:
            raise AgentToolScopeError("Agent did not read the workspace tree")
        return OperationScope(
            incoming_entities=self._incoming_entities,
            candidate_trees=self._trees,
            workspace_tree=self._workspace_tree,
        )


@dataclass(frozen=True, slots=True)
class ValidatedHierarchyOperations:
    """Validated decision plus deterministic IDs for newly created parents."""

    resolution: EntityStructureOperationResolution
    created_parent_ids_by_temporary_id: Mapping[str, str]


def stable_created_parent_id(*, name: str, semantic: str) -> str:
    """Match the existing graph's deterministic virtual-Entity identity scheme."""

    normalized_name = unicodedata.normalize("NFKC", name).casefold()
    name_key = "".join(character for character in normalized_name if character.isalnum())
    identity = f"{name_key}\0{semantic.strip()}"
    return f"entity_virtual_{sha256(identity.encode()).hexdigest()[:16]}"


def validate_hierarchy_operations(
    *,
    resolution: EntityStructureOperationResolution,
    scope: OperationScope,
) -> ValidatedHierarchyOperations:
    """Ensure an Agent can mutate only its batch and retrieved complete trees."""

    errors: list[str] = []
    incoming_ids = scope.incoming_entity_ids
    candidate_ids = scope.candidate_entity_ids
    allowed_ids = incoming_ids | candidate_ids
    handled_incoming: set[str] = set()
    mutated_entity_ids: set[str] = set()
    created_parent_ids: dict[str, str] = {}
    group_operations: list[tuple[GroupEntityOperation, str]] = []
    merge_operations: list[MergeEntityOperation] = []

    def require_allowed(entity_id: str, *, location: str) -> None:
        if entity_id not in allowed_ids:
            errors.append(
                f"{location} references Entity outside the batch and candidate trees: {entity_id}"
            )

    def mark_handled(entity_id: str, *, location: str) -> None:
        if entity_id not in incoming_ids:
            return
        if entity_id in handled_incoming:
            errors.append(f"incoming Entity is handled more than once: {entity_id} ({location})")
        handled_incoming.add(entity_id)

    def mark_mutated(entity_id: str, *, location: str) -> None:
        if entity_id in mutated_entity_ids:
            errors.append(
                f"Entity occurs in conflicting structural operations: {entity_id} ({location})"
            )
        mutated_entity_ids.add(entity_id)

    for operation in resolution.operations:
        if isinstance(operation, MergeEntityOperation):
            merge_operations.append(operation)
            if not set(operation.source_entity_ids).intersection(incoming_ids):
                errors.append("merge must handle at least one incoming Entity")
            for entity_id in operation.source_entity_ids:
                require_allowed(entity_id, location="merge.source_entity_ids")
                mark_handled(entity_id, location="merge")
                mark_mutated(entity_id, location="merge")
        elif isinstance(operation, SeparateEntityOperation):
            if not set(operation.entity_ids).issubset(incoming_ids):
                errors.append("separate may only keep incoming Entities independent")
            for entity_id in operation.entity_ids:
                require_allowed(entity_id, location="separate.entity_ids")
                mark_handled(entity_id, location="separate")
                mark_mutated(entity_id, location="separate")
        elif isinstance(operation, GroupEntityOperation):
            if isinstance(operation.parent, ReusedGroupParent):
                parent_id = operation.parent.parent_entity_id
                if parent_id not in candidate_ids:
                    errors.append("reused group parent must belong to a retrieved candidate tree")
            elif isinstance(operation.parent, CreatedGroupParent):
                if len(operation.children) < 2:
                    errors.append("a newly created group parent must have at least two children")
                parent_id = stable_created_parent_id(
                    name=operation.parent.name,
                    semantic=operation.parent.semantic,
                )
                prior = created_parent_ids.get(operation.parent.temporary_parent_id)
                if prior is not None and prior != parent_id:
                    errors.append("temporary parent ID resolves to inconsistent stable IDs")
                created_parent_ids[operation.parent.temporary_parent_id] = parent_id
            else:  # pragma: no cover - protected by the discriminated Pydantic union.
                errors.append("unsupported group parent")
                continue
            child_ids = [child.child_entity_id for child in operation.children]
            if not set(child_ids).intersection(incoming_ids):
                errors.append("group must handle at least one incoming Entity")
            for entity_id in child_ids:
                require_allowed(entity_id, location="group.children")
                mark_handled(entity_id, location="group")
                mark_mutated(entity_id, location="group")
            group_operations.append((operation, parent_id))

    missing = incoming_ids - handled_incoming
    if missing:
        errors.append(f"Agent omitted incoming Entity IDs: {', '.join(sorted(missing))}")

    for operation, parent_id in group_operations:
        if isinstance(operation.parent, ReusedGroupParent) and parent_id in mutated_entity_ids:
            errors.append(
                "a reused group parent cannot be merged, grouped, or separated in the same batch"
            )

    errors.extend(
        _cycle_validation_errors(
            hierarchy_edges=scope.hierarchy_edges,
            merge_operations=merge_operations,
            group_operations=group_operations,
        )
    )
    if errors:
        raise HierarchyOperationValidationError(errors)
    return ValidatedHierarchyOperations(
        resolution=resolution,
        created_parent_ids_by_temporary_id=created_parent_ids,
    )


def _cycle_validation_errors(
    *,
    hierarchy_edges: Sequence[EntityStructureEdge],
    merge_operations: Sequence[MergeEntityOperation],
    group_operations: Sequence[tuple[GroupEntityOperation, str]],
) -> list[str]:
    """Reject self edges, ancestor merges, and cycles after proposed rewrites."""

    errors: list[str] = []
    representatives: dict[str, str] = {}

    def representative(entity_id: str) -> str:
        parent = representatives.get(entity_id, entity_id)
        while parent != representatives.get(parent, parent):
            parent = representatives[parent]
        return parent

    for operation in merge_operations:
        canonical = operation.canonical_entity_id
        for source in operation.source_entity_ids:
            representatives[source] = canonical
        representatives.setdefault(canonical, canonical)

    rewritten_edges: list[tuple[str, str]] = []
    for edge in hierarchy_edges:
        source = representative(edge.source_entity_id)
        target = representative(edge.target_entity_id)
        if source == target:
            errors.append("merge would collapse an existing hierarchy edge into a self edge")
        else:
            rewritten_edges.append((source, target))
    for operation, parent_id in group_operations:
        target = representative(parent_id)
        for child in operation.children:
            source = representative(child.child_entity_id)
            if source == target:
                errors.append("group would create a self edge")
            else:
                rewritten_edges.append((source, target))

    adjacency: dict[str, set[str]] = {}
    for source, target in rewritten_edges:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set())

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entity_id: str) -> bool:
        if entity_id in visiting:
            return True
        if entity_id in visited:
            return False
        visiting.add(entity_id)
        for target in adjacency.get(entity_id, set()):
            if visit(target):
                return True
        visiting.remove(entity_id)
        visited.add(entity_id)
        return False

    if any(visit(entity_id) for entity_id in adjacency):
        errors.append("operations would create an Entity hierarchy cycle")
    return errors


class EntityHierarchyWorkflowState(TypedDict):
    """Checkpoint-safe coordination state; never put trees or model output here."""

    workspace_id: str
    input_revision: str
    job_id: str
    pending_entity_ids: list[str]
    processed_entity_ids: list[str]
    current_batch_entity_ids: list[str]
    retry_count: int
    phase: Literal["take_batch", "decide", "validate", "repair", "commit", "completed", "failed"]
    operation_id: str | None
    validated_operation_id: str | None
    committed_operation_ids: list[str]


class HierarchyWorkflowResult(BaseModel):
    """Final, checkpoint-safe result returned to the service/API layer."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    input_revision: str
    job_id: str
    processed_entity_ids: list[str]
    committed_operation_ids: list[str]
    pending_entity_ids: list[str]
    phase: Literal["completed"]


@dataclass(slots=True)
class _OperationEnvelope:
    context: HierarchyAgentContext
    resolution: EntityStructureOperationResolution | None
    scope: OperationScope | None
    errors: tuple[str, ...] = field(default_factory=tuple)
    validated: ValidatedHierarchyOperations | None = None


class EntityHierarchyWorkflow:
    """Queue-driven LangGraph coordinator for bounded Entity hierarchy decisions."""

    def __init__(
        self,
        *,
        repository: EntityHierarchyRepository,
        agent: EntityHierarchyAgent,
        checkpointer: Any | None = None,
        batch_size: int = HIERARCHY_BATCH_SIZE,
        max_repair_attempts: int = 1,
    ) -> None:
        if batch_size != HIERARCHY_BATCH_SIZE:
            raise ValueError(f"Entity hierarchy batch_size must remain {HIERARCHY_BATCH_SIZE}")
        if max_repair_attempts != 1:
            raise ValueError("Entity hierarchy workflow supports exactly one automatic repair")
        self._repository = repository
        self._agent = agent
        self._checkpointer = checkpointer
        self._batch_size = batch_size
        self._max_repair_attempts = max_repair_attempts
        self._operations: dict[str, _OperationEnvelope] = {}

    def compile(self) -> Any:
        """Compile the StateGraph; callers may inject a PostgreSQL checkpointer."""

        graph = StateGraph(EntityHierarchyWorkflowState)
        graph.add_node("take_batch", self._take_batch)
        graph.add_node("agent", self._agent_decision)
        graph.add_node("validate", self._validate)
        graph.add_node("repair", self._repair)
        graph.add_node("commit", self._commit)
        graph.add_node("complete", self._complete)
        graph.add_node("fail", self._fail)
        graph.add_edge(START, "take_batch")
        graph.add_conditional_edges(
            "take_batch",
            self._route_after_take_batch,
            {"agent": "agent", "complete": "complete"},
        )
        graph.add_edge("agent", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"agent": "agent", "repair": "repair", "commit": "commit", "fail": "fail"},
        )
        graph.add_edge("repair", "validate")
        graph.add_conditional_edges(
            "commit",
            self._route_after_commit,
            {"take_batch": "take_batch", "agent": "agent"},
        )
        graph.add_edge("complete", END)
        graph.add_edge("fail", END)
        if self._checkpointer is None:
            return graph.compile()
        return graph.compile(checkpointer=self._checkpointer)

    async def run(
        self,
        *,
        workspace_id: str,
        input_revision: str,
        job_id: str,
        pending_entity_ids: Sequence[str],
    ) -> HierarchyWorkflowResult:
        """Run until no new parent Entity remains in the deterministic queue."""

        state: EntityHierarchyWorkflowState = {
            "workspace_id": workspace_id,
            "input_revision": input_revision,
            "job_id": job_id,
            "pending_entity_ids": _stable_unique(pending_entity_ids),
            "processed_entity_ids": [],
            "current_batch_entity_ids": [],
            "retry_count": 0,
            "phase": "take_batch",
            "operation_id": None,
            "validated_operation_id": None,
            "committed_operation_ids": [],
        }
        app = self.compile()
        final_state = await app.ainvoke(
            state,
            config={
                "configurable": {
                    "thread_id": f"entity-hierarchy:{workspace_id}:{job_id}:{input_revision}",
                }
            },
        )
        return HierarchyWorkflowResult(
            workspace_id=final_state["workspace_id"],
            input_revision=final_state["input_revision"],
            job_id=final_state["job_id"],
            processed_entity_ids=final_state["processed_entity_ids"],
            committed_operation_ids=final_state["committed_operation_ids"],
            pending_entity_ids=final_state["pending_entity_ids"],
            phase=final_state["phase"],
        )

    def _take_batch(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        processed = set(state["processed_entity_ids"])
        pending = [
            entity_id for entity_id in state["pending_entity_ids"] if entity_id not in processed
        ]
        batch = pending[: self._batch_size]
        return {
            "pending_entity_ids": pending[len(batch) :],
            "current_batch_entity_ids": batch,
            "retry_count": 0,
            "phase": "decide" if batch else "completed",
            "operation_id": None,
            "validated_operation_id": None,
        }

    async def _agent_decision(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        return await self._run_agent(state=state, repairing=False)

    async def _repair(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        return await self._run_agent(state=state, repairing=True)

    async def _run_agent(
        self,
        *,
        state: EntityHierarchyWorkflowState,
        repairing: bool,
    ) -> dict[str, Any]:
        batch = tuple(state["current_batch_entity_ids"])
        if not batch:
            return {"phase": "completed"}
        retry_count = state["retry_count"] + 1 if repairing else state["retry_count"]
        operation_id = _operation_id(
            job_id=state["job_id"],
            input_revision=state["input_revision"],
            entity_ids=batch,
            retry_count=retry_count,
        )
        nodes = tuple(
            await self._repository.load_entity_nodes(
                workspace_id=state["workspace_id"], entity_ids=batch
            )
        )
        if {node.entity_id for node in nodes} != set(batch) or len(nodes) != len(batch):
            raise EntityHierarchyWorkflowError(
                "repository did not load the exact current Entity batch"
            )
        nodes_by_id = {node.entity_id: node for node in nodes}
        context = HierarchyAgentContext(
            workspace_id=state["workspace_id"],
            input_revision=state["input_revision"],
            job_id=state["job_id"],
            incoming_entities=[nodes_by_id[entity_id] for entity_id in batch],
        )
        tools = BoundedEntityHierarchyTools(
            repository=self._repository,
            workspace_id=state["workspace_id"],
            incoming_entities=context.incoming_entities,
            excluded_entity_ids=[*batch, *state["pending_entity_ids"]],
        )
        previous = self._operations.get(state["operation_id"] or "")
        try:
            if repairing:
                resolution = await self._agent.repair(
                    context=context,
                    tools=tools,
                    previous_resolution=previous.resolution if previous else None,
                    validation_errors=previous.errors if previous else (),
                )
            else:
                resolution = await self._agent.decide(context=context, tools=tools)
            scope = tools.operation_scope()
            envelope = _OperationEnvelope(context=context, resolution=resolution, scope=scope)
        except Exception as exc:
            envelope = _OperationEnvelope(
                context=context,
                resolution=None,
                scope=None,
                errors=(f"Agent decision failed: {exc}",),
            )
        self._operations[operation_id] = envelope
        return {
            "retry_count": retry_count,
            "operation_id": operation_id,
            "validated_operation_id": None,
            "phase": "validate",
        }

    def _validate(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        operation_id = state["operation_id"]
        envelope = self._operations.get(operation_id or "")
        if envelope is None:
            # A checkpoint may outlive this process-local cache.  Re-run the
            # same deterministic operation ID and let the commit be idempotent.
            return {"operation_id": None, "phase": "decide"}
        if envelope.resolution is None or envelope.scope is None:
            return self._invalid_decision_transition(state=state, envelope=envelope)
        try:
            envelope.validated = validate_hierarchy_operations(
                resolution=envelope.resolution,
                scope=envelope.scope,
            )
        except HierarchyOperationValidationError as exc:
            envelope.errors = exc.errors
            return self._invalid_decision_transition(state=state, envelope=envelope)
        return {"validated_operation_id": operation_id, "phase": "commit"}

    def _invalid_decision_transition(
        self,
        *,
        state: EntityHierarchyWorkflowState,
        envelope: _OperationEnvelope,
    ) -> dict[str, Any]:
        if state["retry_count"] < self._max_repair_attempts:
            return {"phase": "repair"}
        return {
            "phase": "failed",
            "operation_id": state["operation_id"],
            "validated_operation_id": None,
        }

    async def _commit(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        operation_id = state["operation_id"]
        envelope = self._operations.get(operation_id or "")
        if envelope is None or envelope.validated is None or envelope.scope is None:
            return {"phase": "decide", "operation_id": None, "validated_operation_id": None}
        request = HierarchyCommitRequest(
            workspace_id=state["workspace_id"],
            input_revision=state["input_revision"],
            job_id=state["job_id"],
            operation_id=operation_id,
            incoming_entity_ids=state["current_batch_entity_ids"],
            candidate_root_entity_ids=list(envelope.scope.candidate_root_entity_ids),
            resolution=envelope.validated.resolution,
            created_parent_ids_by_temporary_id=dict(
                envelope.validated.created_parent_ids_by_temporary_id
            ),
        )
        result = await self._repository.commit_hierarchy_batch(request)
        if result.operation_id != operation_id:
            raise EntityHierarchyWorkflowError(
                "repository returned a mismatched hierarchy operation ID"
            )
        processed = _stable_unique(
            [*state["processed_entity_ids"], *state["current_batch_entity_ids"]]
        )
        forbidden = set(processed)
        pending = _stable_unique(
            [
                *state["pending_entity_ids"],
                *(
                    entity_id
                    for entity_id in result.created_parent_entity_ids
                    if entity_id not in forbidden
                ),
            ]
        )
        committed = _stable_unique([*state["committed_operation_ids"], operation_id])
        return {
            "pending_entity_ids": pending,
            "processed_entity_ids": processed,
            "current_batch_entity_ids": [],
            "retry_count": 0,
            "operation_id": None,
            "validated_operation_id": None,
            "committed_operation_ids": committed,
            "phase": "take_batch",
        }

    @staticmethod
    def _complete(_: EntityHierarchyWorkflowState) -> dict[str, Any]:
        return {"phase": "completed"}

    def _fail(self, state: EntityHierarchyWorkflowState) -> dict[str, Any]:
        envelope = self._operations.get(state["operation_id"] or "")
        reason = "; ".join(envelope.errors) if envelope else "operation cache was unavailable"
        raise EntityHierarchyWorkflowError(
            f"Entity hierarchy Agent could not produce a valid decision after one repair: {reason}"
        )

    @staticmethod
    def _route_after_take_batch(
        state: EntityHierarchyWorkflowState,
    ) -> Literal["agent", "complete"]:
        return "agent" if state["current_batch_entity_ids"] else "complete"

    @staticmethod
    def _route_after_validate(
        state: EntityHierarchyWorkflowState,
    ) -> Literal["agent", "repair", "commit", "fail"]:
        phase = state["phase"]
        if phase == "decide":
            return "agent"
        if phase == "repair":
            return "repair"
        if phase == "commit":
            return "commit"
        return "fail"

    @staticmethod
    def _route_after_commit(state: EntityHierarchyWorkflowState) -> Literal["take_batch", "agent"]:
        return "agent" if state["phase"] == "decide" else "take_batch"


def _operation_id(
    *,
    job_id: str,
    input_revision: str,
    entity_ids: Sequence[str],
    retry_count: int,
) -> str:
    identity = "\0".join((job_id, input_revision, str(retry_count), *sorted(entity_ids)))
    return f"hierarchy:{sha256(identity.encode()).hexdigest()[:24]}"


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    stable: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            stable.append(value)
    return stable
