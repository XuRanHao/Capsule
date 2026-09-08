from __future__ import annotations

from collections.abc import Sequence

import pytest

from capsule.pipeline.entity_hierarchy_workflow import (
    AgentToolScopeError,
    BoundedEntityHierarchyTools,
    CandidateEntityTreeSelection,
    EntityHierarchyTree,
    EntityHierarchyWorkflow,
    HierarchyAgentContext,
    HierarchyCommitRequest,
    HierarchyCommitResult,
    SimilarEntityNodeHit,
    stable_created_parent_id,
)
from capsule.relation_graph import (
    EntityStructureEdge,
    EntityStructureNode,
    EntityStructureOperationResolution,
)


class MemoryHierarchyRepository:
    def __init__(self, nodes: Sequence[EntityStructureNode]) -> None:
        self.nodes = {node.entity_id: node for node in nodes}
        self.search_hits: list[SimilarEntityNodeHit] = []
        self.trees: dict[str, EntityHierarchyTree] = {}
        self.search_calls: list[tuple[str, ...]] = []
        self.tree_calls: list[tuple[str, ...]] = []
        self.commit_requests: list[HierarchyCommitRequest] = []
        self._committed: dict[str, HierarchyCommitResult] = {}

    async def load_entity_nodes(
        self, *, workspace_id: str, entity_ids: Sequence[str]
    ) -> Sequence[EntityStructureNode]:
        del workspace_id
        return [self.nodes[entity_id] for entity_id in entity_ids]

    async def search_similar_entity_nodes(
        self,
        *,
        workspace_id: str,
        incoming_entity_ids: Sequence[str],
        exclude_entity_ids: Sequence[str],
        per_entity_limit: int,
    ) -> Sequence[SimilarEntityNodeHit]:
        del workspace_id, exclude_entity_ids
        assert per_entity_limit == 3
        self.search_calls.append(tuple(incoming_entity_ids))
        return [
            hit for hit in self.search_hits if hit.incoming_entity_id in set(incoming_entity_ids)
        ]

    async def load_complete_entity_trees(
        self, *, workspace_id: str, root_entity_ids: Sequence[str]
    ) -> Sequence[EntityHierarchyTree]:
        del workspace_id
        self.tree_calls.append(tuple(root_entity_ids))
        return [self.trees[root_id] for root_id in root_entity_ids]

    async def load_workspace_tree(self, *, workspace_id: str) -> str:
        assert workspace_id == "workspace_1"
        return "workspace/\n├── characters/\n└── props/"

    async def commit_hierarchy_batch(
        self, request: HierarchyCommitRequest
    ) -> HierarchyCommitResult:
        if request.operation_id in self._committed:
            return self._committed[request.operation_id]
        self.commit_requests.append(request)
        created: list[str] = []
        for operation in request.resolution.operations:
            parent = getattr(operation, "parent", None)
            if getattr(parent, "mode", None) != "create":
                continue
            stable_id = request.created_parent_ids_by_temporary_id[parent.temporary_parent_id]
            if stable_id not in self.nodes:
                self.nodes[stable_id] = EntityStructureNode(
                    entity_id=stable_id,
                    name=parent.name,
                    semantic=parent.semantic,
                )
                created.append(stable_id)
        result = HierarchyCommitResult(
            operation_id=request.operation_id,
            created_parent_entity_ids=created,
        )
        self._committed[request.operation_id] = result
        return result


async def _consume_context(
    tools: BoundedEntityHierarchyTools, context: HierarchyAgentContext
) -> None:
    selection = await tools.retrieve_related_entity_trees(
        [entity.entity_id for entity in context.incoming_entities]
    )
    await tools.read_complete_entity_trees(selection.root_entity_ids)
    await tools.read_workspace_tree()


class GroupThenSeparateAgent:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    async def decide(
        self, *, context: HierarchyAgentContext, tools: BoundedEntityHierarchyTools
    ) -> EntityStructureOperationResolution:
        await _consume_context(tools, context)
        entity_ids = tuple(entity.entity_id for entity in context.incoming_entities)
        self.batches.append(entity_ids)
        if entity_ids == ("entity_a", "entity_b"):
            return EntityStructureOperationResolution.model_validate(
                {
                    "operations": [
                        {
                            "type": "group",
                            "parent": {
                                "mode": "create",
                                "temporary_parent_id": "virtual:story_assets",
                                "name": "故事素材",
                                "semantic": "同一故事的角色与道具素材",
                            },
                            "children": [
                                {
                                    "child_entity_id": "entity_a",
                                    "relation": "角色",
                                    "description": "故事中的角色素材",
                                },
                                {
                                    "child_entity_id": "entity_b",
                                    "relation": "道具",
                                    "description": "故事中的道具素材",
                                },
                            ],
                        }
                    ]
                }
            )
        return EntityStructureOperationResolution.model_validate(
            {"operations": [{"type": "separate", "entity_ids": list(entity_ids)}]}
        )

    async def repair(self, **_: object) -> EntityStructureOperationResolution:
        raise AssertionError("valid Agent output should not enter repair")


class SeparateAgent:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    async def decide(
        self, *, context: HierarchyAgentContext, tools: BoundedEntityHierarchyTools
    ) -> EntityStructureOperationResolution:
        await _consume_context(tools, context)
        entity_ids = tuple(entity.entity_id for entity in context.incoming_entities)
        self.batches.append(entity_ids)
        return EntityStructureOperationResolution.model_validate(
            {"operations": [{"type": "separate", "entity_ids": list(entity_ids)}]}
        )

    async def repair(self, **_: object) -> EntityStructureOperationResolution:
        raise AssertionError("valid Agent output should not enter repair")


class RepairingAgent(SeparateAgent):
    def __init__(self) -> None:
        super().__init__()
        self.repair_calls = 0

    async def decide(
        self, *, context: HierarchyAgentContext, tools: BoundedEntityHierarchyTools
    ) -> EntityStructureOperationResolution:
        await _consume_context(tools, context)
        entity_ids = tuple(entity.entity_id for entity in context.incoming_entities)
        self.batches.append(entity_ids)
        return EntityStructureOperationResolution.model_validate(
            {"operations": [{"type": "separate", "entity_ids": [entity_ids[0]]}]}
        )

    async def repair(
        self,
        *,
        context: HierarchyAgentContext,
        tools: BoundedEntityHierarchyTools,
        previous_resolution: EntityStructureOperationResolution | None,
        validation_errors: Sequence[str],
    ) -> EntityStructureOperationResolution:
        assert previous_resolution is not None
        assert validation_errors
        self.repair_calls += 1
        await _consume_context(tools, context)
        return EntityStructureOperationResolution.model_validate(
            {
                "operations": [
                    {
                        "type": "separate",
                        "entity_ids": [entity.entity_id for entity in context.incoming_entities],
                    }
                ]
            }
        )


def _nodes(*entity_ids: str) -> list[EntityStructureNode]:
    return [
        EntityStructureNode(
            entity_id=entity_id,
            name=entity_id.replace("entity_", "").title(),
            semantic=f"semantic for {entity_id}",
        )
        for entity_id in entity_ids
    ]


@pytest.mark.asyncio
async def test_agent_tools_allow_only_current_batch_and_retrieved_complete_trees() -> None:
    repository = MemoryHierarchyRepository(_nodes("incoming_a", "root_1", "child_1"))
    repository.search_hits = [
        SimilarEntityNodeHit(
            incoming_entity_id="incoming_a",
            entity_id="child_1",
            root_entity_id="root_1",
            score=0.93,
        )
    ]
    repository.trees["root_1"] = EntityHierarchyTree(
        root_entity_id="root_1",
        nodes=[repository.nodes["root_1"], repository.nodes["child_1"]],
        edges=[
            EntityStructureEdge(
                source_entity_id="child_1",
                target_entity_id="root_1",
                relation="child",
                description="child node",
            )
        ],
    )
    tools = BoundedEntityHierarchyTools(
        repository=repository,
        workspace_id="workspace_1",
        incoming_entities=[repository.nodes["incoming_a"]],
    )

    with pytest.raises(AgentToolScopeError, match="exact current batch"):
        await tools.retrieve_related_entity_trees([])

    selection: CandidateEntityTreeSelection = await tools.retrieve_related_entity_trees(
        ["incoming_a"]
    )
    assert selection.root_entity_ids == ["root_1"]
    with pytest.raises(AgentToolScopeError, match="deduplicated candidate roots"):
        await tools.read_complete_entity_trees(["root_not_retrieved"])

    trees = await tools.read_complete_entity_trees(["root_1"])
    await tools.read_workspace_tree()
    scope = tools.operation_scope()

    assert [tree.root_entity_id for tree in trees] == ["root_1"]
    assert scope.candidate_entity_ids == frozenset({"root_1", "child_1"})
    assert repository.tree_calls == [("root_1",)]


@pytest.mark.asyncio
async def test_workflow_reenqueues_only_new_stable_parent_until_hierarchy_converges() -> None:
    repository = MemoryHierarchyRepository(_nodes("entity_a", "entity_b"))
    agent = GroupThenSeparateAgent()
    workflow = EntityHierarchyWorkflow(repository=repository, agent=agent)

    result = await workflow.run(
        workspace_id="workspace_1",
        input_revision="revision_7",
        job_id="job_1",
        pending_entity_ids=["entity_a", "entity_b"],
    )

    expected_parent = stable_created_parent_id(name="故事素材", semantic="同一故事的角色与道具素材")
    assert result.phase == "completed"
    assert result.processed_entity_ids == ["entity_a", "entity_b", expected_parent]
    assert result.pending_entity_ids == []
    assert agent.batches == [("entity_a", "entity_b"), (expected_parent,)]
    assert [request.incoming_entity_ids for request in repository.commit_requests] == [
        ["entity_a", "entity_b"],
        [expected_parent],
    ]
    assert repository.commit_requests[0].created_parent_ids_by_temporary_id == {
        "virtual:story_assets": expected_parent
    }


@pytest.mark.asyncio
async def test_workflow_processes_at_most_ten_entities_per_stategraph_batch() -> None:
    entity_ids = [f"entity_{index}" for index in range(11)]
    repository = MemoryHierarchyRepository(_nodes(*entity_ids))
    agent = SeparateAgent()
    workflow = EntityHierarchyWorkflow(repository=repository, agent=agent)

    result = await workflow.run(
        workspace_id="workspace_1",
        input_revision="revision_1",
        job_id="job_batch_10",
        pending_entity_ids=entity_ids,
    )

    assert result.phase == "completed"
    assert [len(batch) for batch in agent.batches] == [10, 1]
    assert result.processed_entity_ids == entity_ids


@pytest.mark.asyncio
async def test_invalid_agent_output_uses_one_non_human_repair_before_commit() -> None:
    repository = MemoryHierarchyRepository(_nodes("entity_a", "entity_b"))
    agent = RepairingAgent()
    workflow = EntityHierarchyWorkflow(repository=repository, agent=agent)

    result = await workflow.run(
        workspace_id="workspace_1",
        input_revision="revision_4",
        job_id="job_repair",
        pending_entity_ids=["entity_a", "entity_b"],
    )

    assert result.phase == "completed"
    assert agent.repair_calls == 1
    assert len(repository.commit_requests) == 1
