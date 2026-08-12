"""Supervise resident image/text durable-task routes for an ASGI lifespan.

The supervisor owns only the route-loop tasks.  Each worker or scheduler keeps
its existing PostgreSQL/Redis lifecycle, so an application can create this
object at startup and reliably close it before disposing its own dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from capsule.config import Settings, get_settings
from capsule.pipeline.processing_task_service import (
    CpuProcessingTaskScheduler,
    CpuProcessingTaskWorker,
)
from capsule.pipeline.video_task_runtime import ProcessingTaskKind

logger = logging.getLogger(__name__)


class DurableTaskRuntime(Protocol):
    """A route adapter with observable readiness and idempotent cleanup."""

    async def start(self) -> None: ...

    async def run_forever(self) -> None: ...

    async def close(self) -> None: ...


RuntimeFactory = Callable[[], DurableTaskRuntime]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class DurableTaskComponent:
    """A named route runtime that can be rebuilt after an infrastructure failure."""

    name: str
    factory: RuntimeFactory


@dataclass(frozen=True, slots=True)
class DurableTaskHealth:
    """Read-only component state suitable for an app health endpoint."""

    ready: bool
    running: bool
    restart_count: int
    last_error: str | None


@dataclass(slots=True)
class _ManagedComponent:
    spec: DurableTaskComponent
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    runtime: DurableTaskRuntime | None = None
    running: bool = False
    restart_count: int = 0
    last_error: str | None = None


class DurableTaskRuntimeSupervisor:
    """Restart CPU image/text workers and schedulers under one cancellation boundary.

    Video remains intentionally outside the default ASGI set because its MPS
    workload is resident-process work.  Deployments that want it may construct
    a separate supervisor with an explicit ``DurableTaskComponent``.
    """

    def __init__(
        self,
        *,
        components: Sequence[DurableTaskComponent],
        initial_backoff_seconds: float = 0.25,
        max_backoff_seconds: float = 10.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if initial_backoff_seconds < 0 or max_backoff_seconds < initial_backoff_seconds:
            raise ValueError("invalid durable task supervisor backoff")
        names = [component.name for component in components]
        if not all(names) or len(names) != len(set(names)):
            raise ValueError("durable task supervisor component names must be unique and nonempty")
        self._components = [_ManagedComponent(spec=component) for component in components]
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Settings | None = None,
        worker_id: str | None = None,
    ) -> DurableTaskRuntimeSupervisor:
        """Build the default CPU image/text workers and recovery schedulers only."""
        runtime_settings = settings or get_settings()
        identity = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        return cls(
            components=(
                DurableTaskComponent(
                    "cpu-image-worker",
                    lambda: CpuProcessingTaskWorker.for_kind(
                        settings=runtime_settings,
                        task_kind=ProcessingTaskKind.IMAGE,
                        worker_id=f"{identity}-image",
                    ),
                ),
                DurableTaskComponent(
                    "cpu-text-worker",
                    lambda: CpuProcessingTaskWorker.for_kind(
                        settings=runtime_settings,
                        task_kind=ProcessingTaskKind.TEXT,
                        worker_id=f"{identity}-text",
                    ),
                ),
                DurableTaskComponent(
                    "cpu-image-scheduler",
                    lambda: CpuProcessingTaskScheduler.for_kind(
                        settings=runtime_settings,
                        task_kind=ProcessingTaskKind.IMAGE,
                    ),
                ),
                DurableTaskComponent(
                    "cpu-text-scheduler",
                    lambda: CpuProcessingTaskScheduler.for_kind(
                        settings=runtime_settings,
                        task_kind=ProcessingTaskKind.TEXT,
                    ),
                ),
            ),
            initial_backoff_seconds=(runtime_settings.api_embedded_cpu_tasks_restart_seconds),
            max_backoff_seconds=max(
                runtime_settings.api_embedded_cpu_tasks_restart_seconds,
                30.0,
            ),
        )

    async def start(self) -> None:
        """Create route loops and wait until every component has opened its transport."""
        if self._closed:
            raise RuntimeError("durable task runtime supervisor is closed")
        if not self._tasks:
            self._tasks = [
                asyncio.create_task(
                    self._supervise(component),
                    name=f"durable-task-{component.spec.name}",
                )
                for component in self._components
            ]
        await asyncio.gather(*(component.ready_event.wait() for component in self._components))

    async def run_forever(self) -> None:
        """Run all routes until the host cancels this supervisor."""
        await self.start()
        try:
            await asyncio.gather(*self._tasks)
        finally:
            await self.close()

    async def close(self) -> None:
        """Cancel loops and wait for every component to release its own resources."""
        if self._closed:
            return
        self._closed = True
        for component in self._components:
            component.ready_event.clear()
            component.running = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def health_snapshot(self) -> dict[str, DurableTaskHealth]:
        """Return readiness and restart state without exposing runtime objects."""
        return {
            component.spec.name: DurableTaskHealth(
                ready=component.ready_event.is_set(),
                running=component.running,
                restart_count=component.restart_count,
                last_error=component.last_error,
            )
            for component in self._components
        }

    async def _supervise(self, component: _ManagedComponent) -> None:
        backoff = self._initial_backoff_seconds
        while not self._closed:
            runtime: DurableTaskRuntime | None = None
            component.ready_event.clear()
            try:
                runtime = component.spec.factory()
                component.runtime = runtime
                # Queue/group creation is the explicit readiness boundary. A
                # failed start stays unavailable and is rebuilt with backoff.
                await runtime.start()
                component.ready_event.set()
                component.running = True
                await runtime.run_forever()
                if not self._closed:
                    raise RuntimeError("durable task runtime stopped unexpectedly")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                component.ready_event.clear()
                component.running = False
                component.ready_event.clear()
                component.restart_count += 1
                component.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "durable task component failed; rebuilding component=%s",
                    component.spec.name,
                )
                if runtime is not None:
                    await _close_runtime(runtime)
                    runtime = None
                if not self._closed:
                    await self._sleep(backoff)
                    backoff = min(
                        self._max_backoff_seconds,
                        max(backoff * 2, self._initial_backoff_seconds),
                    )
            else:
                return
            finally:
                component.running = False
                if runtime is not None:
                    await _close_runtime(runtime)
                    component.runtime = None


async def _close_runtime(runtime: DurableTaskRuntime) -> None:
    try:
        await runtime.close()
    except Exception:
        logger.exception("durable task component cleanup failed")
