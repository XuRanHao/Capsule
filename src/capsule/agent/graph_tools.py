"""Agent tools for editing the currently selected narrative graph."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from capsule.agent.tools import (
    AgentTool,
    ToolContext,
    ToolExecutionStore,
    ToolHooks,
    ToolRegistry,
)
from capsule.db.repositories import RelationGraphRepository


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListToolOperationsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=20, ge=1, le=100)


class EntityIdArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=128)


class CreateEntityArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=1024)
    entity_type: str = Field(default="", max_length=128)
    description: str = Field(default="", max_length=20_000)


class MergeEntitiesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_entity_id: str = Field(min_length=1, max_length=128)
    source_entity_ids: list[str] = Field(min_length=1, max_length=16)


class SplitEntityPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=1024)
    entity_type: str = Field(default="", max_length=128)
    description: str = Field(default="", max_length=20_000)
    asset_ids: list[str] = Field(default_factory=list, max_length=500)


class SplitEntityArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1, max_length=128)
    parts: list[SplitEntityPart] = Field(min_length=2, max_length=2)


class MoveAssetArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1, max_length=128)
    entity_id: str = Field(min_length=1, max_length=128)
    binding_role: str = Field(default="reference", max_length=128)
    description: str = Field(default="", max_length=20_000)


class CreateParentRelationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    child_entity_id: str = Field(min_length=1, max_length=128)
    parent_name: str = Field(min_length=1, max_length=1024)
    parent_entity_type: str = Field(default="", max_length=128)
    parent_description: str = Field(default="", max_length=20_000)
    relation_description: str = Field(default="", max_length=20_000)


class RemoveRelationArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_id: str = Field(min_length=1, max_length=128)


def build_graph_tool_registry(
    repository: RelationGraphRepository,
    execution_store: ToolExecutionStore | None = None,
    hooks: ToolHooks | None = None,
) -> ToolRegistry:
    """Register only tools that operate inside the user-selected graph."""

    def graph_id(context: ToolContext) -> str:
        if not context.graph_id:
            raise ValueError("a graph must be selected before using graph tools")
        return context.graph_id

    async def load_context(_: BaseModel, context: ToolContext) -> dict[str, Any]:
        return await repository.load_current_graph_context(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
        )

    async def get_entity(args: EntityIdArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.get_entity_detail(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            entity_id=args.entity_id,
        )

    async def list_relations(
        args: EntityIdArguments, context: ToolContext
    ) -> list[dict[str, Any]]:
        return await repository.list_entity_relations(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            entity_id=args.entity_id,
        )

    async def list_operations(
        args: ListToolOperationsArguments, context: ToolContext
    ) -> list[dict[str, Any]]:
        if execution_store is None:
            raise ValueError("tool execution history is not configured")
        return await execution_store.list_for_thread(
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            thread_id=context.thread_id,
            limit=args.limit,
        )

    async def create(args: CreateEntityArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.create_entity(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            name=args.name,
            entity_type=args.entity_type,
            description=args.description,
        )

    async def merge(args: MergeEntitiesArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.merge_entities(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            target_entity_id=args.target_entity_id,
            source_entity_ids=args.source_entity_ids,
        )

    async def split(args: SplitEntityArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.split_entity(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            entity_id=args.entity_id,
            parts=[part.model_dump() for part in args.parts],
        )

    async def delete(args: EntityIdArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.delete_entity(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            entity_id=args.entity_id,
        )

    async def move_asset(args: MoveAssetArguments, context: ToolContext) -> dict[str, Any]:
        return await repository.move_asset_to_entity(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            asset_id=args.asset_id,
            entity_id=args.entity_id,
            binding_role=args.binding_role,
            description=args.description,
        )

    async def create_parent(
        args: CreateParentRelationArguments, context: ToolContext
    ) -> dict[str, Any]:
        return await repository.create_parent_relation(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            child_entity_id=args.child_entity_id,
            parent_name=args.parent_name,
            parent_entity_type=args.parent_entity_type,
            parent_description=args.parent_description,
            relation_description=args.relation_description,
        )

    async def remove_relation(
        args: RemoveRelationArguments, context: ToolContext
    ) -> dict[str, Any]:
        return await repository.remove_relation(
            workspace_id=context.workspace_id,
            graph_id=graph_id(context),
            relation_id=args.relation_id,
        )

    read_tools = [
        AgentTool(
            name="load_current_graph_context",
            description="读取当前选中图谱的实体、关系和叙事背景。",
            args_schema=_NoArguments,
            handler=load_context,
            timeout_seconds=3.0,
            required_permission="graph:read",
        ),
        AgentTool(
            name="get_entity_detail",
            description="读取指定实体及其关联素材。",
            args_schema=EntityIdArguments,
            handler=get_entity,
            timeout_seconds=3.0,
            required_permission="graph:read",
        ),
        AgentTool(
            name="list_entity_relations",
            description="读取指定实体的全部关系。",
            args_schema=EntityIdArguments,
            handler=list_relations,
            timeout_seconds=3.0,
            required_permission="graph:read",
        ),
        AgentTool(
            name="list_recent_tool_operations",
            description="读取当前用户在本次会话中的工具调用记录。",
            args_schema=ListToolOperationsArguments,
            handler=list_operations,
            timeout_seconds=3.0,
            required_permission="graph:read",
        ),
    ]
    write_tools = [
        AgentTool(
            name="create_entity",
            description="在当前图谱中创建逻辑实体。",
            args_schema=CreateEntityArguments,
            handler=create,
            timeout_seconds=5.0,
            required_permission="graph:write",
        ),
        AgentTool(
            name="merge_entities",
            description="合并实体并迁移其关联素材和关系。",
            args_schema=MergeEntitiesArguments,
            handler=merge,
            timeout_seconds=10.0,
            requires_confirmation=True,
            required_permission="graph:destructive",
        ),
        AgentTool(
            name="split_entity",
            description="拆分实体并按指定清单重新分配素材。",
            args_schema=SplitEntityArguments,
            handler=split,
            timeout_seconds=10.0,
            requires_confirmation=True,
            required_permission="graph:destructive",
        ),
        AgentTool(
            name="delete_entity",
            description="删除当前图谱中的实体及其绑定和关系。",
            args_schema=EntityIdArguments,
            handler=delete,
            timeout_seconds=5.0,
            requires_confirmation=True,
            required_permission="graph:destructive",
        ),
        AgentTool(
            name="move_asset_to_entity",
            description="将图谱中的素材移动到指定实体。",
            args_schema=MoveAssetArguments,
            handler=move_asset,
            timeout_seconds=5.0,
            required_permission="graph:write",
        ),
        AgentTool(
            name="create_parent_relation",
            description="创建父实体并建立父子层级关系。",
            args_schema=CreateParentRelationArguments,
            handler=create_parent,
            timeout_seconds=5.0,
            required_permission="graph:write",
        ),
        AgentTool(
            name="remove_relation",
            description="删除当前图谱中的一条实体关系。",
            args_schema=RemoveRelationArguments,
            handler=remove_relation,
            timeout_seconds=5.0,
            requires_confirmation=True,
            required_permission="graph:destructive",
        ),
    ]
    return ToolRegistry(
        read_tools + write_tools,
        execution_store=execution_store,
        hooks=hooks,
    )
