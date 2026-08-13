"""Reliable processing-task runtime built around PostgreSQL and Redis Streams.

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
from enum import StrEnum
from typing import Any, Protocol


class ProcessingTaskKind(StrEnum):
    """Stable worker routing class, persisted in transport messages."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ResourceClass(StrEnum):
    """Worker-pool resource requirement declared by each task message."""

    CPU = "cpu"
    MPS_VIDEO = "mps_video"
    IO = "io"


def _resource_class_for(kind: ProcessingTaskKind) -> ResourceClass:
    return (
        ResourceClass.MPS_VIDEO
        if kind in {ProcessingTaskKind.VIDEO, ProcessingTaskKind.AUDIO}
        else ResourceClass.CPU
    )


def _stable_task_id(
    job_id: str,
    source_file_id: str,
    generation: int,
    *,
    task_kind: ProcessingTaskKind,
    result_version: int,
) -> str:
    """Return a repeatable identifier for one processor/source generation."""
    if task_kind is ProcessingTaskKind.VIDEO:
        # This byte sequence is the pre-generalization id contract.  Preserve it
        # exactly because existing PostgreSQL/Redis video rows already use it.
        raw = f"{job_id}\x1f{source_file_id}\x1f{generation}".encode()
        return f"video-{hashlib.sha256(raw).hexdigest()[:24]}"
    # New kinds match the durable uniqueness scope: a source generation, kind,
    # and result schema version. Job retries/reimports must not create aliases.
    raw = f"{source_file_id}\x1f{generation}\x1f{task_kind.value}\x1f{result_version}".encode()
    return f"{task_kind.value}-{hashlib.sha256(raw).hexdigest()[:24]}"


@dataclass(frozen=True, slots=True)
class ProcessingTaskMessage:
    """The complete, versioned transport request for one source generation."""

    job_id: str
    workspace_id: str
    source_file_id: str
    generation: int
    attempt: int = 0
    task_id: str | None = None
    result_version: int = 1
    source_uri: str = ""
    task_kind: ProcessingTaskKind = ProcessingTaskKind.VIDEO
    processor_version: int = 1
    resource_class: ResourceClass | None = None
    message_schema_version: int = 1
    dispatch_version: int = 0
    route_key: str = "mps_video"
    legacy_wire: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.job_id or not self.workspace_id or not self.source_file_id:
            raise ValueError("job_id, workspace_id and source_file_id are required")
        if self.generation < 1:
            raise ValueError("generation must be positive")
        if self.attempt < 0:
            raise ValueError("attempt cannot be negative")
        if self.result_version < 1:
            raise ValueError("result_version must be positive")
        if self.processor_version < 1:
            raise ValueError("processor_version must be positive")
        if self.message_schema_version < 1:
            raise ValueError("message_schema_version must be positive")
        if self.dispatch_version < 0:
            raise ValueError("dispatch_version cannot be negative")
        if not self.route_key:
            raise ValueError("route_key is required")
        if self.resource_class is None:
            object.__setattr__(self, "resource_class", _resource_class_for(self.task_kind))
        if self.task_id is None:
            object.__setattr__(
                self,
                "task_id",
                _stable_task_id(
                    self.job_id,
                    self.source_file_id,
                    self.generation,
                    task_kind=self.task_kind,
                    result_version=self.result_version,
                ),
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
            "task_kind": self.task_kind.value,
            "processor_version": str(self.processor_version),
            "resource_class": (self.resource_class or _resource_class_for(self.task_kind)).value,
            "message_schema_version": str(self.message_schema_version),
            "dispatch_version": str(self.dispatch_version),
            "route_key": self.route_key,
        }

    @classmethod
    def from_fields(cls, fields: Mapping[str, str | bytes]) -> ProcessingTaskMessage:
        def value(name: str, default: str | None = None) -> str:
            raw = fields.get(name, default)
            if raw is None:
                raise ValueError(f"processing task message is missing {name}")
            return raw.decode() if isinstance(raw, bytes) else str(raw)

        contract_fields = frozenset(
            {
                "task_kind",
                "processor_version",
                "resource_class",
                "message_schema_version",
                "dispatch_version",
                "route_key",
            }
        )
        present_contract_fields = contract_fields.intersection(fields)
        if present_contract_fields and present_contract_fields != contract_fields:
            missing = ", ".join(sorted(contract_fields - present_contract_fields))
            raise ValueError(f"processing task message has partial routing contract: {missing}")
        legacy_wire = not present_contract_fields
        task_kind = ProcessingTaskKind(value("task_kind", ProcessingTaskKind.VIDEO.value))
        return cls(
            task_id=value("task_id") or None,
            job_id=value("job_id"),
            workspace_id=value("workspace_id"),
            source_file_id=value("source_file_id"),
            generation=int(value("generation")),
            attempt=int(value("attempt", "0")),
            result_version=int(value("result_version", "1")),
            source_uri=value("source_uri", ""),
            # Pre-generalization video entries omitted these fields.  Defaulting
            # here lets their original task ids and handling remain unchanged.
            task_kind=task_kind,
            processor_version=int(value("processor_version", "1")),
            resource_class=ResourceClass(
                value("resource_class", _resource_class_for(task_kind).value)
            ),
            message_schema_version=int(value("message_schema_version", "1")),
            dispatch_version=int(value("dispatch_version", "0")),
            route_key=value("route_key", "mps_video"),
            legacy_wire=legacy_wire,
        )


@dataclass(frozen=True, slots=True)
class ProcessingTaskDelivery:
    message: ProcessingTaskMessage
    receipt: str


@dataclass(frozen=True, slots=True)
class ProcessingTaskLease:
    """A PostgreSQL-fenced ownership token; ``attempt`` is never client supplied."""

    task_id: str
    source_file_id: str
    source_generation: int
    attempt: int
    worker_id: str
    result_version: int
    task_kind: ProcessingTaskKind = ProcessingTaskKind.VIDEO
    resource_class: ResourceClass = ResourceClass.MPS_VIDEO
    route_key: str = "mps_video"
    processor_version: int = 1
    # Pre-token callers can still construct a lease for unit tests, but leases
    # returned by a repository claim always carry this fresh database fence.
    lease_token: str = ""


@dataclass(frozen=True, slots=True)
class ProcessingTaskProgress:
    completed_units: int = 0
    total_units: int | None = None
    detail: Mapping[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProcessingTaskResult:
    """References persisted results; payload bytes do not travel in Redis."""

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


class ProcessingTaskQueue(Protocol):
    async def publish(self, message: ProcessingTaskMessage) -> str: ...

    async def receive(self) -> ProcessingTaskDelivery: ...

    async def acknowledge(self, delivery: ProcessingTaskDelivery) -> None: ...

    async def retry(
        self,
        delivery: ProcessingTaskDelivery,
        *,
        next_attempt: int,
        delay_seconds: float,
    ) -> None: ...

    async def route_dlq(self, delivery: ProcessingTaskDelivery, *, error: str) -> None: ...

    async def quarantine(self, delivery: ProcessingTaskDelivery, *, error: str) -> None: ...


class ProcessingTaskRepository(Protocol):
    """PostgreSQL boundary.  Every mutating method must fence on the supplied lease."""

    async def inspect_message_contract(self, message: ProcessingTaskMessage) -> bool | None:
        """Classify a normalized delivery against its durable task identity.

        ``True`` permits claiming, ``False`` denotes a persisted identity mismatch,
        and ``None`` denotes an orphan with no matching durable task row.
        """
        ...

    async def claim_attempt(
        self,
        message: ProcessingTaskMessage,
        *,
        worker_id: str,
        receipt: str,
    ) -> ProcessingTaskLease | None: ...

    async def can_ack_unclaimed(self, message: ProcessingTaskMessage) -> bool: ...

    async def heartbeat(self, lease: ProcessingTaskLease) -> bool: ...

    async def record_progress(
        self,
        lease: ProcessingTaskLease,
        progress: ProcessingTaskProgress,
    ) -> bool: ...

    async def complete(self, lease: ProcessingTaskLease, result: ProcessingTaskResult) -> bool: ...

    async def schedule_retry(
        self,
        lease: ProcessingTaskLease,
        *,
        error: str,
        retry_at: float,
    ) -> bool: ...

    async def fail(self, lease: ProcessingTaskLease, *, error: str) -> bool: ...

    async def mark_dlq_published(self, message: ProcessingTaskMessage) -> None: ...


class ProcessingTaskProcessor(Protocol):
    async def process(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
    ) -> ProcessingTaskResult: ...


# Compatibility aliases keep the existing video service, database adapter and
# worker implementation source-compatible while new text/image workers use the
# generic names above.
VideoTaskMessage = ProcessingTaskMessage
VideoTaskDelivery = ProcessingTaskDelivery
VideoTaskLease = ProcessingTaskLease
VideoTaskProgress = ProcessingTaskProgress
VideoTaskResult = ProcessingTaskResult
VideoTaskQueue = ProcessingTaskQueue
VideoTaskRepository = ProcessingTaskRepository
VideoTaskProcessor = ProcessingTaskProcessor


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
        expected_route_key: str = "mps_video",
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id is required")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if not expected_route_key:
            raise ValueError("expected_route_key is required")
        self._queue = queue
        self._repository = repository
        self._processor = processor
        self._worker_id = worker_id
        self._retry_policy = retry_policy or RetryPolicy()
        self._heartbeat_seconds = heartbeat_seconds
        self._expected_route_key = expected_route_key
        self._clock = clock

    async def handle_delivery(self, delivery: VideoTaskDelivery) -> str:
        """Return the final delivery outcome, including ``quarantined``."""
        try:
            # Imported lazily because the registry uses task value classes from
            # this module; it is a pure contract check before any DB mutation.
            from capsule.pipeline.processing_task_registry import validate_message_contract

            validate_message_contract(
                delivery.message,
                expected_route_key=self._expected_route_key,
            )
        except ValueError as exc:
            # Queue quarantine owns the XADD -> XACK order. If XADD fails, it
            # raises and the transport delivery deliberately remains pending.
            await self._queue.quarantine(delivery, error=_quarantine_error(exc))
            return "quarantined"

        # The database is authoritative for a task's workspace/source/generation
        # identity.  This read-only classification must occur before claim so a
        # valid wire message cannot mutate an unrelated durable task.
        durable_contract = await self._repository.inspect_message_contract(delivery.message)
        if durable_contract is False:
            await self._queue.quarantine(delivery, error="durable_contract_mismatch")
            return "quarantined"
        if durable_contract is None:
            await self._queue.quarantine(delivery, error="orphan_task")
            return "quarantined"

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
        quarantine_stream: str | None = None,
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
        self._quarantine_stream = quarantine_stream or f"{stream}:quarantine"
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
                delivery = await self._decode_or_quarantine(messages[0])
                if delivery is not None:
                    return delivery
                continue
            response = await client.xreadgroup(
                self._group,
                self._consumer,
                {self._stream: ">"},
                count=1,
                block=1_000,
            )
            if response:
                delivery = await self._decode_or_quarantine(response[0][1][0])
                if delivery is not None:
                    return delivery

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

    async def quarantine(self, delivery: VideoTaskDelivery, *, error: str) -> None:
        """Durably quarantine then ACK exactly once; never ACK on write failure."""
        await self._write_quarantine(
            receipt=delivery.receipt,
            fields=delivery.message.to_fields(),
            error=error,
        )
        await self.acknowledge(delivery)

    def _required_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("RedisVideoTaskQueue has not been started")
        return self._client

    async def _decode_or_quarantine(
        self,
        message: tuple[str | bytes, Mapping[str, str | bytes]],
    ) -> VideoTaskDelivery | None:
        receipt, fields = message
        normalized_receipt = receipt.decode() if isinstance(receipt, bytes) else str(receipt)
        try:
            return VideoTaskDelivery(VideoTaskMessage.from_fields(fields), normalized_receipt)
        except (TypeError, ValueError) as exc:
            await self._write_quarantine(
                receipt=normalized_receipt,
                fields=fields,
                error=_quarantine_error(exc),
            )
            await self._required_client().xack(self._stream, self._group, normalized_receipt)
            return None

    async def _write_quarantine(
        self,
        *,
        receipt: str,
        fields: Mapping[str, str | bytes],
        error: str,
    ) -> None:
        safe_fields = {
            str(key): value.decode() if isinstance(value, bytes) else str(value)
            for key, value in fields.items()
            if key != "source_uri"
        }
        safe_fields.update(
            {
                "source_receipt": receipt,
                "error": error[:2_000],
                "quarantine_reason": "invalid_task_contract",
            }
        )
        await self._required_client().xadd(self._quarantine_stream, safe_fields)


def _quarantine_error(exc: BaseException) -> str:
    """Avoid copying untrusted raw fields (notably source URIs) into error text."""
    return f"processing task contract rejected: {type(exc).__name__}"


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


# Runtime/transport aliases complete the gradual migration: existing imports
# retain their video names while new workers can depend on generic contracts.
ProcessingTaskRuntime = VideoTaskRuntime
RedisProcessingTaskQueue = RedisVideoTaskQueue
ProcessingTaskRecoveryRepository = VideoTaskRecoveryRepository
ProcessingTaskRecoveryScheduler = VideoTaskRecoveryScheduler
