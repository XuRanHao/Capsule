"""LangGraph runtime for the conversational narrative-graph Agent.

The database conversation repository imports small Agent contracts.  Keep this
package initializer lazy so importing that repository cannot recursively import
the full runtime back from its partially initialized module.
"""

from __future__ import annotations

from typing import Any

from capsule.agent.contracts import AgentRequest, AgentResponse, PlanDecision, ToolCall

__all__ = [
    "AgentRequest",
    "AgentResponse",
    "AgentRuntime",
    "AgentPlanner",
    "AgentTool",
    "InMemoryMemoryStore",
    "ModelAgentPlanner",
    "NullMemoryStore",
    "PlanDecision",
    "ReadyPlanner",
    "ToolCall",
    "ToolCancellationToken",
    "ToolContext",
    "ToolHooks",
    "ToolLockProvider",
    "ToolRegistry",
    "RedisToolLockProvider",
    "RetryableToolError",
    "create_agent_runtime",
    "build_graph_tool_registry",
]


def __getattr__(name: str) -> Any:
    if name in {"AgentPlanner", "ReadyPlanner"}:
        from capsule.agent.graph import AgentPlanner, ReadyPlanner

        return {"AgentPlanner": AgentPlanner, "ReadyPlanner": ReadyPlanner}[name]
    if name == "ModelAgentPlanner":
        from capsule.agent.model_planner import ModelAgentPlanner

        return ModelAgentPlanner
    if name in {"AgentRuntime", "create_agent_runtime"}:
        from capsule.agent.runtime import AgentRuntime, create_agent_runtime

        return {"AgentRuntime": AgentRuntime, "create_agent_runtime": create_agent_runtime}[name]
    if name in {"InMemoryMemoryStore", "NullMemoryStore"}:
        from capsule.agent.memory import InMemoryMemoryStore, NullMemoryStore

        return {
            "InMemoryMemoryStore": InMemoryMemoryStore,
            "NullMemoryStore": NullMemoryStore,
        }[name]
    if name == "build_graph_tool_registry":
        from capsule.agent.graph_tools import build_graph_tool_registry

        return build_graph_tool_registry
    if name in {
        "AgentTool",
        "RedisToolLockProvider",
        "RetryableToolError",
        "ToolCancellationToken",
        "ToolContext",
        "ToolHooks",
        "ToolLockProvider",
        "ToolRegistry",
    }:
        from capsule.agent import tools

        return getattr(tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
