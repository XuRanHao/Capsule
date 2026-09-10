from unittest.mock import AsyncMock

import pytest

from capsule.agent.contracts import ToolCall
from capsule.agent.graph_tools import build_graph_tool_registry
from capsule.agent.tools import ToolContext


def _context() -> ToolContext:
    return ToolContext(
        user_id="user",
        workspace_id="workspace",
        thread_id="thread",
        graph_id="graph",
        state={},
    )


def test_graph_tool_registry_exposes_only_current_graph_tools() -> None:
    repository = AsyncMock()
    registry = build_graph_tool_registry(repository)

    descriptions = registry.describe()
    names = {description["name"] for description in descriptions}
    assert names == {
        "load_current_graph_context",
        "get_entity_detail",
        "list_entity_relations",
        "create_entity",
        "merge_entities",
        "split_entity",
        "delete_entity",
        "move_asset_to_entity",
        "create_parent_relation",
        "remove_relation",
    }
    assert {
        description["name"]
        for description in descriptions
        if description["requires_confirmation"]
    } == {
        "create_entity",
        "merge_entities",
        "split_entity",
        "delete_entity",
        "move_asset_to_entity",
        "create_parent_relation",
        "remove_relation",
    }


@pytest.mark.asyncio
async def test_graph_write_tool_requires_confirmation_before_repository_call() -> None:
    repository = AsyncMock()
    registry = build_graph_tool_registry(repository)

    result = await registry.execute(
        ToolCall(name="create_entity", arguments={"name": "角色"}),
        context=_context(),
    )

    assert result.error_code == "confirmation_required"
    assert result.needs_confirmation is True
    repository.create_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_read_tool_injects_selected_graph_context() -> None:
    repository = AsyncMock()
    repository.get_entity_detail.return_value = {"entity_id": "entity"}
    registry = build_graph_tool_registry(repository)

    result = await registry.execute(
        ToolCall(name="get_entity_detail", arguments={"entity_id": "entity"}),
        context=_context(),
    )

    assert result.ok is True
    repository.get_entity_detail.assert_awaited_once_with(
        workspace_id="workspace",
        graph_id="graph",
        entity_id="entity",
    )
