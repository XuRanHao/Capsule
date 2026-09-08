"""Model adapter for one bounded Entity-hierarchy Agent decision.

The adapter deliberately owns no persistence capability.  It reads the exact
candidate context permitted by :class:`BoundedEntityHierarchyTools`, converts
the complete trees to the existing structure-model contract, and delegates the
decision (including the model client's JSON repair) to the existing client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from capsule.pipeline.entity_hierarchy_workflow import (
    BoundedEntityHierarchyTools,
    EntityHierarchyTree,
    HierarchyAgentContext,
)
from capsule.relation_graph import EntityStructureOperationResolution


@runtime_checkable
class EntityStructureOperationsModel(Protocol):
    """The existing model-client boundary used for hierarchy decisions."""

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: Mapping[str, Any],
        incoming_entities: Sequence[Mapping[str, Any]],
        workspace_tree: str,
        repair_guidance: str | None = None,
    ) -> EntityStructureOperationResolution:
        """Return the existing validated structure-operation schema."""


class EntityHierarchyAgentAdapter:
    """Use the existing structure Prompt with an explicitly bounded read chain.

    This is intentionally a deterministic tool-call adapter rather than a
    provider-specific function-calling loop.  It can later be replaced by
    native provider tool calling without changing the workflow or persistence
    contracts, because those layers only depend on ``EntityHierarchyAgent``.
    """

    def __init__(self, *, model: EntityStructureOperationsModel) -> None:
        self._model = model

    async def decide(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
    ) -> EntityStructureOperationResolution:
        """Read the bounded context in order, then request one decision."""

        return await self._request_structure_decision(context=context, tools=tools)

    async def repair(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
        previous_resolution: EntityStructureOperationResolution | None,
        validation_errors: Sequence[str],
    ) -> EntityStructureOperationResolution:
        """Retry the existing model Prompt against newly hydrated bounded context.

        ``generate_entity_structure_operations`` already performs up to three
        schema/structure repairs internally.  The workflow-level repair is a
        fresh bounded invocation after backend validation, with no human-review
        branch and no database write capability exposed to the model.
        """

        del previous_resolution
        return await self._request_structure_decision(
            context=context,
            tools=tools,
            repair_guidance="\n".join(validation_errors),
        )

    async def _request_structure_decision(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
        repair_guidance: str | None = None,
    ) -> EntityStructureOperationResolution:
        selection = await tools.retrieve_related_entity_trees(
            [entity.entity_id for entity in context.incoming_entities]
        )
        trees = await tools.read_complete_entity_trees(selection.root_entity_ids)
        workspace_tree = await tools.read_workspace_tree()
        request: dict[str, Any] = {
            "current_graph": _current_graph(trees),
            "incoming_entities": [
                entity.model_dump(mode="json") for entity in context.incoming_entities
            ],
            "workspace_tree": workspace_tree,
        }
        if repair_guidance:
            request["repair_guidance"] = repair_guidance
        return await self._model.generate_entity_structure_operations(**request)


def _current_graph(trees: Sequence[EntityHierarchyTree]) -> dict[str, list[dict[str, Any]]]:
    """Flatten every retrieved complete tree into the legacy structure contract."""

    return {
        "nodes": [
            node.model_dump(mode="json")
            for tree in trees
            for node in tree.nodes
        ],
        "edges": [
            edge.model_dump(mode="json")
            for tree in trees
            for edge in tree.edges
        ],
    }
