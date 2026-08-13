"""Dispatch the shared MPS route by durable processing task kind."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskLease,
    ProcessingTaskMessage,
    ProcessingTaskProcessor,
    ProcessingTaskProgress,
    ProcessingTaskRepository,
    ProcessingTaskResult,
)


class MediaRepositoryBackend(ProcessingTaskRepository, Protocol):
    async def recover_timed_out(self, *, now: float) -> None: ...

    async def dispatchable_messages(self, *, now: float) -> list[ProcessingTaskMessage]: ...

    async def mark_published(self, message: ProcessingTaskMessage) -> None: ...

    async def due_failed_dlq(self, *, now: float) -> list[tuple[ProcessingTaskMessage, str]]: ...


class MediaTaskProcessor:
    def __init__(self, processors: Mapping[ProcessingTaskKind, ProcessingTaskProcessor]) -> None:
        self._processors = dict(processors)

    async def process(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
    ) -> ProcessingTaskResult:
        processor = self._processors.get(message.task_kind)
        if processor is None:
            raise ValueError(f"shared MPS Worker cannot process {message.task_kind.value}")
        return await processor.process(message, lease, report_progress)


class MediaTaskRepository:
    """Delegate repository fences to the fixed identity for each media kind."""

    def __init__(self, repositories: Mapping[ProcessingTaskKind, MediaRepositoryBackend]) -> None:
        self._repositories = dict(repositories)

    def _message_repository(self, message: ProcessingTaskMessage) -> MediaRepositoryBackend:
        repository = self._repositories.get(message.task_kind)
        if repository is None:
            raise ValueError(f"no repository for {message.task_kind.value}")
        return repository

    def _lease_repository(self, lease: ProcessingTaskLease) -> MediaRepositoryBackend:
        repository = self._repositories.get(lease.task_kind)
        if repository is None:
            raise ValueError(f"no repository for {lease.task_kind.value}")
        return repository

    async def inspect_message_contract(self, message: ProcessingTaskMessage) -> bool | None:
        return await self._message_repository(message).inspect_message_contract(message)

    async def claim_attempt(
        self, message: ProcessingTaskMessage, *, worker_id: str, receipt: str
    ) -> ProcessingTaskLease | None:
        return await self._message_repository(message).claim_attempt(
            message, worker_id=worker_id, receipt=receipt
        )

    async def can_ack_unclaimed(self, message: ProcessingTaskMessage) -> bool:
        return await self._message_repository(message).can_ack_unclaimed(message)

    async def heartbeat(self, lease: ProcessingTaskLease) -> bool:
        return await self._lease_repository(lease).heartbeat(lease)

    async def record_progress(
        self, lease: ProcessingTaskLease, progress: ProcessingTaskProgress
    ) -> bool:
        return await self._lease_repository(lease).record_progress(lease, progress)

    async def complete(self, lease: ProcessingTaskLease, result: ProcessingTaskResult) -> bool:
        return await self._lease_repository(lease).complete(lease, result)

    async def schedule_retry(
        self, lease: ProcessingTaskLease, *, error: str, retry_at: float
    ) -> bool:
        return await self._lease_repository(lease).schedule_retry(
            lease, error=error, retry_at=retry_at
        )

    async def fail(self, lease: ProcessingTaskLease, *, error: str) -> bool:
        return await self._lease_repository(lease).fail(lease, error=error)

    async def mark_dlq_published(self, message: ProcessingTaskMessage) -> None:
        await self._message_repository(message).mark_dlq_published(message)

    async def recover_timed_out(self, *, now: float) -> None:
        for repository in self._repositories.values():
            await repository.recover_timed_out(now=now)

    async def dispatchable_messages(self, *, now: float) -> list[ProcessingTaskMessage]:
        messages: list[ProcessingTaskMessage] = []
        for repository in self._repositories.values():
            messages.extend(await repository.dispatchable_messages(now=now))
        return messages

    async def mark_published(self, message: ProcessingTaskMessage) -> None:
        await self._message_repository(message).mark_published(message)

    async def due_failed_dlq(self, *, now: float) -> list[tuple[ProcessingTaskMessage, str]]:
        messages: list[tuple[ProcessingTaskMessage, str]] = []
        for repository in self._repositories.values():
            messages.extend(await repository.due_failed_dlq(now=now))
        return messages
