from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from capsule.pipeline.entity_hierarchy_agent import EntityHierarchyAgentAdapter
from capsule.pipeline.entity_hierarchy_workflow import (
    CandidateEntityTreeSelection,
    EntityHierarchyTree,
    HierarchyAgentContext,
)
from capsule.relation_graph import (
    EntityStructureEdge,
    EntityStructureNode,
    EntityStructureOperationResolution,
)


class RecordingReadTools:
    """A read-only stand-in proving the adapter cannot invoke a write tool."""

    def __init__(self, *, events: list[str], trees: Sequence[EntityHierarchyTree]) -> None:
        self._events = events
        self._trees = tuple(trees)

    async def retrieve_related_entity_trees(
        self, entity_ids: Sequence[str]
    ) -> CandidateEntityTreeSelection:
        self._events.append(f"retrieve:{','.join(entity_ids)}")
        return CandidateEntityTreeSelection(
            root_entity_ids=[tree.root_entity_id for tree in self._trees]
        )

    async def read_complete_entity_trees(
        self, root_entity_ids: Sequence[str]
    ) -> tuple[EntityHierarchyTree, ...]:
        self._events.append(f"read_trees:{','.join(root_entity_ids)}")
        return self._trees

    async def read_workspace_tree(self) -> str:
        self._events.append("read_workspace")
        return "workspace/\n├── characters/\n└── props/"


class RecordingStructureModel:
    def __init__(self, *, events: list[str]) -> None:
        self._events = events
        self.calls: list[dict[str, object]] = []

    async def generate_entity_structure_operations(
        self,
        *,
        current_graph: Mapping[str, Any],
        incoming_entities: Sequence[Mapping[str, Any]],
        workspace_tree: str,
        repair_guidance: str | None = None,
    ) -> EntityStructureOperationResolution:
        self._events.append("model")
        self.calls.append(
            {
                "current_graph": current_graph,
                "incoming_entities": list(incoming_entities),
                "workspace_tree": workspace_tree,
                "repair_guidance": repair_guidance,
            }
        )
        return EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "separate",
                        "entity_ids": [entity["entity_id"] for entity in incoming_entities],
                    }
                ]
            }
        )


def _node(entity_id: str) -> EntityStructureNode:
    return EntityStructureNode(
        entity_id=entity_id,
        name=entity_id.replace("entity_", "").title(),
        semantic=f"semantic for {entity_id}",
    )


def _context(*entity_ids: str) -> HierarchyAgentContext:
    return HierarchyAgentContext(
        workspace_id="workspace_1",
        input_revision="revision_1",
        job_id="job_1",
        incoming_entities=[_node(entity_id) for entity_id in entity_ids],
    )


def _complete_trees() -> tuple[EntityHierarchyTree, ...]:
    root = _node("entity_root")
    child = _node("entity_child")
    other_root = _node("entity_other_root")
    return (
        EntityHierarchyTree(
            root_entity_id=root.entity_id,
            nodes=[root, child],
            edges=[
                EntityStructureEdge(
                    source_entity_id=child.entity_id,
                    target_entity_id=root.entity_id,
                    relation="part_of",
                    description="child belongs to the complete root tree",
                )
            ],
        ),
        EntityHierarchyTree(
            root_entity_id=other_root.entity_id,
            nodes=[other_root],
            edges=[],
        ),
    )


@pytest.mark.asyncio
async def test_decide_reads_bounded_tools_in_order_and_passes_complete_trees() -> None:
    events: list[str] = []
    trees = _complete_trees()
    tools = RecordingReadTools(events=events, trees=trees)
    model = RecordingStructureModel(events=events)
    agent = EntityHierarchyAgentAdapter(model=model)

    result = await agent.decide(
        context=_context("entity_incoming_a", "entity_incoming_b"),
        tools=tools,  # type: ignore[arg-type]
    )

    assert [operation.type for operation in result.operations] == ["separate"]
    assert events == [
        "retrieve:entity_incoming_a,entity_incoming_b",
        "read_trees:entity_root,entity_other_root",
        "read_workspace",
        "model",
    ]
    call = model.calls[0]
    assert call["incoming_entities"] == [
        _node("entity_incoming_a").model_dump(mode="json"),
        _node("entity_incoming_b").model_dump(mode="json"),
    ]
    assert call["current_graph"] == {
        "nodes": [
            _node("entity_root").model_dump(mode="json"),
            _node("entity_child").model_dump(mode="json"),
            _node("entity_other_root").model_dump(mode="json"),
        ],
        "edges": [trees[0].edges[0].model_dump(mode="json")],
    }
    assert call["workspace_tree"] == "workspace/\n├── characters/\n└── props/"


@pytest.mark.asyncio
async def test_repair_rehydrates_only_read_context_then_reuses_structure_model() -> None:
    events: list[str] = []
    tools = RecordingReadTools(events=events, trees=_complete_trees())
    model = RecordingStructureModel(events=events)
    agent = EntityHierarchyAgentAdapter(model=model)
    context = _context("entity_incoming")

    await agent.repair(
        context=context,
        tools=tools,  # type: ignore[arg-type]
        previous_resolution=EntityStructureOperationResolution(operations=[]),
        validation_errors=("Agent omitted incoming Entity IDs",),
    )

    assert events == [
        "retrieve:entity_incoming",
        "read_trees:entity_root,entity_other_root",
        "read_workspace",
        "model",
    ]
    assert not hasattr(tools, "commit_hierarchy_batch")
    assert model.calls[0]["repair_guidance"] == "Agent omitted incoming Entity IDs"
