"""Reliable long-video task runtime built around PostgreSQL ownership and Redis Streams.

Redis is deliberately only a transport in this module.  A ``VideoTaskRepository``
implementation must persist leases, attempts, progress, retry state and results in
the Capsule database inside transactions.  In particular, ``complete``/``fail``
must fence on ``(task_id, attempt, worker_id)`` and return only after their database
transaction commits.  The runtime acknowledges a Streams entry afterwards.

Operational construction lives in ``video_task_service`` so this transport and
state-machine module stays independent from Capsule's parsing implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


def _stable_task_id(job_id: str, source_file_id: str, generation: int) -> str:
    """Return a repeatable identifier for all deliveries of one source generation."""
    raw = f"{job_id}\x1f{source_file_id}\x1f{generation}".encode()
    return f"video-{hashlib.sha256(raw).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class VideoTaskMessage:
    """The complete, versioned message for one whole-video processing request."""

    job_id: str
    workspace_id: str
    source_file_id: str
    generation: int
    attempt: int = 0
    task_id: str | None = None
    result_version: int = 1
    source_uri: str = ""

    def __post_init__(self) -> None:
        if not self.job_id or not self.workspace_id or not self.source_file_id:
            raise ValueError("job_id, workspace_id and source_file_id are required")
        if self.generation < 1:
            raise ValueError("generation must be positive")
        if self.attempt < 0:
            raise ValueError("attempt cannot be negative")
        if self.result_version < 1:
            raise ValueError("result_version must be positive")
        if self.task_id is None:
            object.__setattr__(
                self,
                "task_id",
                _stable_task_id(self.job_id, self.source_file_id, self.generation),
            )

    def to_fields(self) -> dict[str, str]:
        """Encode only primitive Redis Stream field values."""
        return {
            "task_id": self.task_id or "",
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "source_file_id": self.source_file_id,
            "generation": str(self.generation),
            "attempt": str(self.attempt),
            "result_version": str(self.result_version),
            "source_uri": self.source_uri,
        }

    @classmethod
    def from_fields(cls, fields: Mapping[str, str | bytes]) -> VideoTaskMessage:
        def value(name: str, default: str | None = None) -> str:
            raw = fields.get(name, default)
            if raw is None:
                raise ValueError(f"video task message is missing {name}")
            return raw.decode() if isinstance(raw, bytes) else str(raw)

        return cls(
            task_id=value("task_id") or None,
            job_id=value("job_id"),
            workspace_id=value("workspace_id"),
            source_file_id=value("source_file_id"),
            generation=int(value("generation")),
            attempt=int(value("attempt", "0")),
            result_version=int(value("result_version", "1")),
            source_uri=value("source_uri", ""),
        )


@dataclass(frozen=True, slots=True)
class VideoTaskDelivery:
    message: VideoTaskMessage
    receipt: str


@dataclass(frozen=True, slots=True)
class VideoTaskLease:
    """A PostgreSQL-fenced ownership token; ``attempt`` is never client supplied."""

    task_id: str
    source_file_id: str
    source_generation: int
    attempt: int
    worker_id: str
    result_version: int


@dataclass(frozen=True, slots=True)
class VideoTaskProgress:
    completed_units: int = 0
    total_units: int | None = None
    detail: Mapping[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VideoTaskResult:
    """References persisted derived media; payload bytes do not travel in Redis."""

    result_ref: str
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    retry_delays_seconds: tuple[float, ...] = (5.0, 30.0, 300.0)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not self.retry_delays_seconds or any(delay < 0 for delay in self.retry_delays_seconds):
            raise ValueError("retry_delays_seconds must contain non-negative delays")

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the delay after a failed one-based database attempt."""
        index = min(max(attempt - 1, 0), len(self.retry_delays_seconds) - 1)
        return self.retry_delays_seconds[index]


class VideoTaskQueue(Protocol):
    async def publish(self, message: VideoTaskMessage) -> str: ...

    async def receive(self) -> VideoTaskDelivery: ...

    async def acknowledge(self, delivery: VideoTaskDelivery) -> None: ...

    async def retry(
        self,
        delivery: VideoTaskDelivery,
        *,
        next_attempt: int,
        delay_seconds: float,
    ) -> None: ...

    async def route_dlq(self, delivery: VideoTaskDelivery, *, error: str) -> None: ...


class VideoTaskRepository(Protocol):
    """PostgreSQL boundary.  Every mutating method must fence on the supplied lease."""

    async def claim_attempt(
        self,
        message: VideoTaskMessage,
        *,
        worker_id: str,
        receipt: str,
    ) -> VideoTaskLease | None: ...

    async def can_ack_unclaimed(self, message: VideoTaskMessage) -> bool: ...

    async def heartbeat(self, lease: VideoTaskLease) -> bool: ...

    async def record_progress(self, lease: VideoTaskLease, progress: VideoTaskProgress) -> bool: ...

    async def complete(self, lease: VideoTaskLease, result: VideoTaskResult) -> bool: ...

    async def schedule_retry(
        self,
        lease: VideoTaskLease,
        *,
        error: str,
        retry_at: float,
    ) -> bool: ...

    async def fail(self, lease: VideoTaskLease, *, error: str) -> bool: ...

    async def mark_dlq_published(self, message: VideoTaskMessage) -> None: ...


class VideoTaskProcessor(Protocol):
    async def process(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        report_progress: Callable[[VideoTaskProgress], Awaitable[None]],
    ) -> VideoTaskResult: ...


class LeaseLostError(RuntimeError):
    """Raised when a stale worker attempts to report progress or finish work."""


class VideoTaskRuntime:
    """Process one delivery with database-before-ACK finalization semantics."""

    def __init__(
        self,
        *,
        queue: VideoTaskQueue,
        repository: VideoTaskRepository,
        processor: VideoTaskProcessor,
        worker_id: str,
        retry_policy: RetryPolicy | None = None,
        heartbeat_seconds: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id is required")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._queue = queue
        self._repository = repository
        self._processor = processor
        self._worker_id = worker_id
        self._retry_policy = retry_policy or RetryPolicy()
        self._heartbeat_seconds = heartbeat_seconds
        self._clock = clock

    async def handle_delivery(self, delivery: VideoTaskDelivery) -> str:
        """Return ``completed``, ``duplicate``, ``retry_scheduled`` or ``dlq``."""
        lease = await self._repository.claim_attempt(
            delivery.message,
            worker_id=self._worker_id,
            receipt=delivery.receipt,
        )
        if lease is None:
            if await self._repository.can_ack_unclaimed(delivery.message):
                await self._queue.acknowledge(delivery)
                return "duplicate"
            # XAUTOCLAIM can observe a long-running delivery before PostgreSQL
            # has invalidated its active lease.  Leaving it pending avoids an
            # early ACK; the authoritative attempt or the recovery scheduler
            # will converge it later.
            return "deferred"

        stopped = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_loop(lease, stopped, lease_lost)
        )
        processor = asyncio.create_task(
            self._processor.process(
                delivery.message,
                lease,
                lambda progress: self._report_progress(lease, progress),
            )
        )
        try:
            done, _ = await asyncio.wait(
                {processor, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done and lease_lost.is_set():
                processor.cancel()
                await asyncio.gather(processor, return_exceptions=True)
                await self._queue.acknowledge(delivery)
                return "fenced"
            result = await processor
            committed = await self._repository.complete(lease, result)
            # A false completion means another valid attempt committed or fenced us.
            # It is still safe to remove this duplicate transport delivery.
            await self._queue.acknowledge(delivery)
            return "completed" if committed else "duplicate"
        except LeaseLostError:
            await self._queue.acknowledge(delivery)
            return "duplicate"
        except Exception as exc:
            return await self._handle_failure(delivery, lease, str(exc))
        finally:
            stopped.set()
            if not processor.done():
                processor.cancel()
                await asyncio.gather(processor, return_exceptions=True)
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def _report_progress(self, lease: VideoTaskLease, progress: VideoTaskProgress) -> None:
        # A first progress update also establishes liveness before a processor can
        # spend a long interval inside native FFmpeg/MPS code.
        if not await self._repository.heartbeat(lease):
            raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")
        if not await self._repository.record_progress(lease, progress):
            raise LeaseLostError(f"task lease lost: {lease.task_id}/{lease.attempt}")

    async def _heartbeat_loop(
        self,
        lease: VideoTaskLease,
        stopped: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        while not stopped.is_set():
            try:
                await asyncio.wait_for(stopped.wait(), timeout=self._heartbeat_seconds)
                return
            except TimeoutError:
                pass
            try:
                if not await self._repository.heartbeat(lease):
                    lease_lost.set()
                    return
            except Exception:
                # The processor's next progress/commit is still fenced by PostgreSQL.
                # A transient heartbeat failure must not make the worker self-ACK work.
                pass

    async def _handle_failure(
        self,
        delivery: VideoTaskDelivery,
        lease: VideoTaskLease,
        error: str,
    ) -> str:
        bounded_error = error[:2_000] or "video processor failed"
        if lease.attempt >= self._retry_policy.max_attempts:
            if not await self._repository.fail(lease, error=bounded_error):
                if await self._repository.can_ack_unclaimed(delivery.message):
                    await self._queue.acknowledge(delivery)
                    return "duplicate"
                return "deferred"
            await self._queue.route_dlq(delivery, error=bounded_error)
            await self._repository.mark_dlq_published(delivery.message)
            await self._queue.acknowledge(delivery)
            return "dlq"

        delay = self._retry_policy.delay_for_attempt(lease.attempt)
        retry_at = self._clock() + delay
        if not await self._repository.schedule_retry(
            lease,
            error=bounded_error,
            retry_at=retry_at,
        ):
            await self._queue.acknowledge(delivery)
            return "duplicate"
        await self._queue.retry(
            delivery,
            next_attempt=lease.attempt + 1,
            delay_seconds=delay,
        )
        return "retry_scheduled"


class RedisVideoTaskQueue:
    """Redis Streams adapter with PEL recovery, delayed retry and a separate DLQ."""

    def __init__(
        self,
        *,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str,
        dlq_stream: str,
        claim_idle_ms: int = 30_000,
        delayed_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if claim_idle_ms < 1:
            raise ValueError("claim_idle_ms must be positive")
        self._redis_url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._dlq_stream = dlq_stream
        self._delayed_key = delayed_key or f"{stream}:delayed"
        self._claim_idle_ms = claim_idle_ms
        self._client = client

    async def start(self) -> None:
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self._redis_url, decode_responses=True)
        try:
            await self._client.xgroup_create(self._stream, self._group, id="0-0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def publish(self, message: VideoTaskMessage) -> str:
        return str(await self._required_client().xadd(self._stream, message.to_fields()))

    async def receive(self) -> VideoTaskDelivery:
        client = self._required_client()
        while True:
            await self.release_due_retries()
            claimed = await client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                min_idle_time=self._claim_idle_ms,
                start_id="0-0",
                count=1,
            )
            messages = claimed[1] if len(claimed) > 1 else []
            if messages:
                return self._delivery(messages[0])
            response = await client.xreadgroup(
                self._group,
                self._consumer,
                {self._stream: ">"},
                count=1,
                block=1_000,
            )
            if response:
                return self._delivery(response[0][1][0])

    async def acknowledge(self, delivery: VideoTaskDelivery) -> None:
        await self._required_client().xack(self._stream, self._group, delivery.receipt)

    async def retry(
        self,
        delivery: VideoTaskDelivery,
        *,
        next_attempt: int,
        delay_seconds: float,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")
        message = replace(delivery.message, attempt=next_attempt)
        payload = json.dumps(message.to_fields(), sort_keys=True, separators=(",", ":"))
        # ACK only after Redis has durably recorded the delayed delivery.  A crash
        # before ACK leaves a harmless duplicate in the PEL, fenced by PostgreSQL.
        await self._required_client().zadd(
            self._delayed_key,
            {payload: time.time() + delay_seconds},
        )
        await self.acknowledge(delivery)

    async def release_due_retries(self, *, limit: int = 100) -> int:
        """Return delayed entries to the stream; duplicates are intentional-safe."""
        client = self._required_client()
        now = time.time()
        payloads = await client.zrangebyscore(self._delayed_key, "-inf", now, start=0, num=limit)
        moved = 0
        for payload in payloads:
            decoded = payload.decode() if isinstance(payload, bytes) else payload
            fields = json.loads(decoded)
            await client.xadd(self._stream, fields)
            await client.zrem(self._delayed_key, payload)
            moved += 1
        return moved

    async def route_dlq(self, delivery: VideoTaskDelivery, *, error: str) -> None:
        fields = delivery.message.to_fields()
        fields.update({"source_receipt": delivery.receipt, "error": error[:2_000]})
        await self._required_client().xadd(self._dlq_stream, fields)

    def _required_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("RedisVideoTaskQueue has not been started")
        return self._client

    @staticmethod
    def _delivery(message: tuple[str | bytes, Mapping[str, str | bytes]]) -> VideoTaskDelivery:
        receipt, fields = message
        normalized_receipt = receipt.decode() if isinstance(receipt, bytes) else str(receipt)
        return VideoTaskDelivery(VideoTaskMessage.from_fields(fields), normalized_receipt)


class VideoTaskRecoveryRepository(Protocol):
    """Optional scheduler API implemented by the PostgreSQL adapter."""

    async def recover_timed_out(self, *, now: float) -> None: ...

    async def dispatchable_messages(self, *, now: float) -> list[VideoTaskMessage]: ...

    async def mark_published(self, message: VideoTaskMessage) -> None: ...

    async def due_failed_dlq(
        self, *, now: float
    ) -> list[tuple[VideoTaskMessage, str]]: ...

    async def mark_dlq_published(self, message: VideoTaskMessage) -> None: ...


class VideoTaskRecoveryScheduler:
    """Run periodic timeout recovery and database-backed retry redispatching."""

    def __init__(
        self,
        *,
        queue: VideoTaskQueue,
        repository: VideoTaskRecoveryRepository,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._queue = queue
        self._repository = repository
        self._clock = clock

    async def run_once(self) -> int:
        now = self._clock()
        await self._repository.recover_timed_out(now=now)
        published = 0
        for message in await self._repository.dispatchable_messages(now=now):
            # Publishing before marking leaves a duplicate on a scheduler crash;
            # the database claim fence turns it into an ACK-only delivery.
            await self._queue.publish(message)
            await self._repository.mark_published(message)
            published += 1
        for message, error in await self._repository.due_failed_dlq(now=now):
            await self._queue.route_dlq(
                VideoTaskDelivery(message=message, receipt=f"pg:{message.task_id}"),
                error=error,
            )
            await self._repository.mark_dlq_published(message)
            published += 1
        return published
