"""Lifecycle and recovery contracts for the durable task runtime supervisor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from capsule.pipeline.durable_task_supervisor import (
    DurableTaskComponent,
    DurableTaskRuntimeSupervisor,
)


@dataclass
class _Runtime:
    name: str
    events: list[str]
    started: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    close_calls: int = 0

    async def start(self) -> None:
        self.events.append(f"ready:{self.name}")
        self.started.set()

    async def run_forever(self) -> None:
        self.events.append(f"run:{self.name}")
        try:
            await self.released.wait()
        finally:
            self.events.append(f"stop:{self.name}")

    async def close(self) -> None:
        self.close_calls += 1
        self.released.set()


async def test_supervisor_waits_for_every_component_readiness_and_closes_once() -> None:
    events: list[str] = []
    scheduler = _Runtime("scheduler", events)
    worker = _Runtime("worker", events)
    supervisor = DurableTaskRuntimeSupervisor(
        components=(
            DurableTaskComponent("scheduler", lambda: scheduler),
            DurableTaskComponent("worker", lambda: worker),
        )
    )

    await supervisor.start()
    await supervisor.start()
    await supervisor.close()
    await supervisor.close()

    assert events.count("ready:scheduler") == events.count("ready:worker") == 1
    assert scheduler.close_calls == worker.close_calls == 1
    snapshot = supervisor.health_snapshot()
    assert not snapshot["scheduler"].ready and not snapshot["scheduler"].running
    assert not snapshot["worker"].ready and not snapshot["worker"].running


async def test_supervisor_rebuilds_a_failed_component_after_backoff() -> None:
    events: list[str] = []
    attempts = 0
    recovered = _Runtime("recovered", events)

    class FailingRuntime(_Runtime):
        async def run_forever(self) -> None:
            self.events.append("run:failing")
            raise ConnectionError("redis disappeared")

    def factory() -> _Runtime:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return FailingRuntime("failing", events)
        return recovered

    supervisor = DurableTaskRuntimeSupervisor(
        components=(DurableTaskComponent("cpu-image-worker", factory),),
        initial_backoff_seconds=0,
        max_backoff_seconds=0,
    )

    await supervisor.start()
    await asyncio.wait_for(recovered.started.wait(), timeout=1)
    snapshot = supervisor.health_snapshot()["cpu-image-worker"]
    await supervisor.close()

    assert attempts == 2
    assert snapshot.ready and snapshot.running
    assert snapshot.restart_count == 1
    assert snapshot.last_error == "ConnectionError: redis disappeared"
    assert recovered.close_calls == 1


async def test_supervisor_rejects_restart_after_lifespan_shutdown() -> None:
    supervisor = DurableTaskRuntimeSupervisor(components=())

    await supervisor.close()

    with pytest.raises(RuntimeError, match="is closed"):
        await supervisor.start()


async def test_supervisor_run_forever_cleans_up_on_cancellation() -> None:
    events: list[str] = []
    runtime = _Runtime("worker", events)
    supervisor = DurableTaskRuntimeSupervisor(
        components=(DurableTaskComponent("worker", lambda: runtime),)
    )
    running = asyncio.create_task(supervisor.run_forever())
    await asyncio.wait_for(runtime.started.wait(), timeout=1)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert runtime.close_calls == 1
