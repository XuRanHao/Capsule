"""Controlled local benchmark for Agent tool governance and slot scheduling.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_agent_tools.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from typing import Any

from pydantic import BaseModel, Field

from capsule.agent.contracts import ToolCall
from capsule.agent.tools import AgentTool, ToolContext, ToolExecutionResult, ToolRegistry

VALIDATION_RUNS = 100
LATENCY_ROUNDS = 50
HANDLER_DELAY_SECONDS = 0.02
SLOT_CAPACITY = 4


class ResourceArguments(BaseModel):
    resource_id: str = Field(min_length=1)
    value: str = Field(min_length=1)


class CalculationArguments(BaseModel):
    value: str = Field(min_length=1)


def _context(*, permissions: frozenset[str]) -> ToolContext:
    return ToolContext(
        user_id="benchmark-user",
        workspace_id="benchmark-workspace",
        thread_id="benchmark-thread",
        graph_id=None,
        state={},
        granted_permissions=permissions,
    )


def _calls() -> list[ToolCall]:
    """Ten calls with three conflict chains and independent calculations."""

    return [
        ToolCall(name="write_resource", arguments={"resource_id": "a", "value": "a"}),
        ToolCall(name="read_resource", arguments={"resource_id": "b", "value": "b"}),
        ToolCall(name="calculate", arguments={"value": "calculate-1"}),
        ToolCall(name="read_resource", arguments={"resource_id": "a", "value": "a-read"}),
        ToolCall(name="delete_resource", arguments={"resource_id": "b", "value": "b-delete"}),
        ToolCall(name="write_resource", arguments={"resource_id": "c", "value": "c"}),
        ToolCall(name="normalize", arguments={"value": "normalize-1"}),
        ToolCall(name="read_resource", arguments={"resource_id": "c", "value": "c-read"}),
        ToolCall(name="write_resource", arguments={"resource_id": "d", "value": "d"}),
        ToolCall(name="calculate", arguments={"value": "calculate-2"}),
    ]


def _registry(*, handler_delay_seconds: float) -> ToolRegistry:
    async def resource_handler(args: BaseModel, _: ToolContext) -> dict[str, str]:
        validated = ResourceArguments.model_validate(args)
        if handler_delay_seconds:
            await asyncio.sleep(handler_delay_seconds)
        return {"value": validated.value}

    async def calculation_handler(args: BaseModel, _: ToolContext) -> dict[str, str]:
        validated = CalculationArguments.model_validate(args)
        if handler_delay_seconds:
            await asyncio.sleep(handler_delay_seconds)
        return {"value": validated.value}

    return ToolRegistry(
        [
            AgentTool(
                name="read_resource",
                description="benchmark resource read",
                args_schema=ResourceArguments,
                handler=resource_handler,
                concurrency_mode="parallel",
                resource_id_field="resource_id",
                resource_operation="read",
                required_permission="graph:read",
            ),
            AgentTool(
                name="write_resource",
                description="benchmark resource write",
                args_schema=ResourceArguments,
                handler=resource_handler,
                concurrency_mode="parallel",
                resource_id_field="resource_id",
                resource_operation="write",
                required_permission="graph:write",
            ),
            AgentTool(
                name="delete_resource",
                description="benchmark resource delete",
                args_schema=ResourceArguments,
                handler=resource_handler,
                concurrency_mode="parallel",
                resource_id_field="resource_id",
                resource_operation="write",
                required_permission="graph:write",
            ),
            AgentTool(
                name="calculate",
                description="benchmark pure calculation",
                args_schema=CalculationArguments,
                handler=calculation_handler,
                concurrency_mode="parallel",
            ),
            AgentTool(
                name="normalize",
                description="benchmark pure normalization",
                args_schema=CalculationArguments,
                handler=calculation_handler,
                concurrency_mode="parallel",
            ),
        ],
        slot_capacity=SLOT_CAPACITY,
    )


async def _count_results(
    calls: list[ToolCall],
    *,
    context: ToolContext,
    registry: ToolRegistry,
) -> list[ToolExecutionResult]:
    return [await registry.execute(call, context=context) for call in calls]


def _percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


async def _latencies(*, concurrent: bool) -> list[float]:
    registry = _registry(handler_delay_seconds=HANDLER_DELAY_SECONDS)
    context = _context(permissions=frozenset({"graph:write"}))
    values: list[float] = []
    for _ in range(LATENCY_ROUNDS):
        started = time.perf_counter()
        calls = _calls()
        if concurrent:
            results = await registry.execute_batch(calls, context=context)
        else:
            results = await _count_results(calls, context=context, registry=registry)
        if not all(result.ok for result in results):
            raise RuntimeError("benchmark legal call unexpectedly failed")
        values.append((time.perf_counter() - started) * 1_000)
    return values


async def run() -> dict[str, Any]:
    schema_registry = _registry(handler_delay_seconds=0)
    schema_results = await _count_results(
        [
            ToolCall(name="read_resource", arguments={"resource_id": "", "value": "x"})
            for _ in range(VALIDATION_RUNS)
        ],
        context=_context(permissions=frozenset({"graph:read"})),
        registry=schema_registry,
    )
    permission_registry = _registry(handler_delay_seconds=0)
    permission_results = await _count_results(
        [
            ToolCall(
                name="write_resource",
                arguments={"resource_id": "restricted", "value": "x"},
            )
            for _ in range(VALIDATION_RUNS)
        ],
        context=_context(permissions=frozenset({"graph:read"})),
        registry=permission_registry,
    )
    success_registry = _registry(handler_delay_seconds=0)
    success_results = await _count_results(
        [
            ToolCall(name="calculate", arguments={"value": str(index)})
            for index in range(VALIDATION_RUNS)
        ],
        context=_context(permissions=frozenset({"graph:write"})),
        registry=success_registry,
    )
    serial = await _latencies(concurrent=False)
    concurrent = await _latencies(concurrent=True)
    return {
        "environment": {
            "type": "controlled_local",
            "validation_runs": VALIDATION_RUNS,
            "latency_rounds": LATENCY_ROUNDS,
            "handler_delay_ms": HANDLER_DELAY_SECONDS * 1_000,
            "slot_capacity": SLOT_CAPACITY,
            "calls_per_round": len(_calls()),
            "tool_types": 5,
        },
        "governance": {
            "schema_intercept_rate": sum(
                result.error_code == "invalid_arguments" for result in schema_results
            )
            / VALIDATION_RUNS,
            "permission_intercept_rate": sum(
                result.error_code == "permission_denied" for result in permission_results
            )
            / VALIDATION_RUNS,
            "legal_tool_success_rate": sum(result.ok for result in success_results)
            / VALIDATION_RUNS,
        },
        "group_latency_ms": {
            "serial_p95": _percentile_95(serial),
            "slot_concurrent_p95": _percentile_95(concurrent),
            "serial_mean": statistics.fmean(serial),
            "slot_concurrent_mean": statistics.fmean(concurrent),
        },
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))
