"""Pressure-test the processing-task runtime without PostgreSQL or Asset writes.

The benchmark uses a real Redis Streams server and the production queue/runtime,
but replaces the durable repository and task processor with deterministic in-memory
implementations.  It therefore measures transport, contract validation, dispatch,
runtime scheduling and ACK behavior without writing any database row or Asset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from redis.asyncio import Redis

from capsule.config import Settings
from capsule.pipeline.video_task_runtime import (
    RedisVideoTaskQueue,
    VideoTaskLease,
    VideoTaskMessage,
    VideoTaskProgress,
    VideoTaskResult,
    VideoTaskRuntime,
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    profile: str
    task_count: int
    malformed_count: int
    concurrency: int
    processor_delay_ms: float
    wall_seconds: float
    producer_seconds: float
    publish_throughput_per_second: float
    completion_throughput_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    peak_inflight: int
    average_cpu_cores: float
    process_peak_rss_mb: float
    redis_peak_delta_mb: float
    completed_count: int
    quarantined_count: int
    final_pending_count: int
    error_count: int


class InMemoryTaskRepository:
    """A fenced task fact store used only by this no-database benchmark."""

    def __init__(self) -> None:
        self._messages: dict[str, VideoTaskMessage] = {}
        self._states: dict[str, str] = {}
        self._attempts: dict[str, int] = {}
        self._tokens: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(self, message: VideoTaskMessage) -> None:
        task_id = _task_id(message)
        async with self._lock:
            self._messages[task_id] = message
            self._states[task_id] = "queued"
            self._attempts[task_id] = 0

    async def inspect_message_contract(self, message: VideoTaskMessage) -> bool | None:
        stored = self._messages.get(_task_id(message))
        return None if stored is None else stored == message

    async def claim_attempt(
        self,
        message: VideoTaskMessage,
        *,
        worker_id: str,
        receipt: str,
    ) -> VideoTaskLease | None:
        del receipt
        task_id = _task_id(message)
        async with self._lock:
            if self._states.get(task_id) != "queued":
                return None
            attempt = self._attempts[task_id] + 1
            token = uuid4().hex
            self._attempts[task_id] = attempt
            self._tokens[task_id] = token
            self._states[task_id] = "processing"
        resource_class = message.resource_class
        if resource_class is None:  # pragma: no cover - message post-init invariant
            raise RuntimeError("benchmark message has no resource class")
        return VideoTaskLease(
            task_id=task_id,
            source_file_id=message.source_file_id,
            source_generation=message.generation,
            attempt=attempt,
            worker_id=worker_id,
            result_version=message.result_version,
            task_kind=message.task_kind,
            resource_class=resource_class,
            route_key=message.route_key,
            processor_version=message.processor_version,
            lease_token=token,
        )

    async def can_ack_unclaimed(self, message: VideoTaskMessage) -> bool:
        return self._states.get(_task_id(message)) == "completed"

    async def heartbeat(self, lease: VideoTaskLease) -> bool:
        return self._owns(lease)

    async def record_progress(
        self,
        lease: VideoTaskLease,
        progress: VideoTaskProgress,
    ) -> bool:
        del progress
        return self._owns(lease)

    async def complete(self, lease: VideoTaskLease, result: VideoTaskResult) -> bool:
        del result
        async with self._lock:
            if not self._owns(lease):
                return False
            self._states[lease.task_id] = "completed"
            return True

    async def schedule_retry(
        self,
        lease: VideoTaskLease,
        *,
        error: str,
        retry_at: float,
    ) -> bool:
        del lease, error, retry_at
        return False

    async def fail(self, lease: VideoTaskLease, *, error: str) -> bool:
        del lease, error
        return False

    async def mark_dlq_published(self, message: VideoTaskMessage) -> None:
        del message

    def _owns(self, lease: VideoTaskLease) -> bool:
        return (
            self._states.get(lease.task_id) == "processing"
            and self._tokens.get(lease.task_id) == lease.lease_token
        )


class DelayedNoopProcessor:
    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    async def process(
        self,
        message: VideoTaskMessage,
        lease: VideoTaskLease,
        report_progress: Any,
    ) -> VideoTaskResult:
        del report_progress
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        return VideoTaskResult(
            result_ref=f"memory://{lease.task_id}",
            metadata={"source_file_id": message.source_file_id},
        )


class RunTracker:
    def __init__(self) -> None:
        self.published_at: dict[str, float] = {}
        self.latencies_ms: list[float] = []
        self.inflight = 0
        self.peak_inflight = 0
        self.completed = 0
        self.errors: list[str] = []

    def published(self, task_id: str, at: float) -> None:
        self.published_at[task_id] = at
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)

    def finished(self, task_id: str, at: float) -> None:
        started = self.published_at[task_id]
        self.latencies_ms.append((at - started) * 1000)
        self.inflight -= 1
        self.completed += 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="control-plane")
    parser.add_argument("--task-count", type=int, default=5_000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--processor-delay-ms", type=float, default=0.0)
    parser.add_argument("--malformed-rate", type=float, default=0.0)
    parser.add_argument("--redis-url", default=Settings().redis_url)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task_count < 1:
        parser.error("--task-count must be positive")
    if any(value < 1 for value in args.concurrency):
        parser.error("--concurrency values must be positive")
    if args.processor_delay_ms < 0:
        parser.error("--processor-delay-ms cannot be negative")
    if not 0 <= args.malformed_rate < 1:
        parser.error("--malformed-rate must be in [0, 1)")
    return args


async def _run_once(
    *,
    redis_url: str,
    profile: str,
    task_count: int,
    concurrency: int,
    processor_delay_ms: float,
    malformed_rate: float,
) -> BenchmarkResult:
    suffix = uuid4().hex
    prefix = f"capsule:benchmark:processing-task:{suffix}"
    stream = f"{prefix}:stream"
    group = f"benchmark-{suffix}"
    dlq_stream = f"{prefix}:dlq"
    quarantine_stream = f"{prefix}:quarantine"
    delayed_key = f"{prefix}:delayed"
    admin = Redis.from_url(redis_url, decode_responses=True)
    queue = RedisVideoTaskQueue(
        redis_url=redis_url,
        stream=stream,
        group=group,
        consumer=f"worker-{suffix}",
        dlq_stream=dlq_stream,
        quarantine_stream=quarantine_stream,
        delayed_key=delayed_key,
        claim_idle_ms=60_000,
    )
    repository = InMemoryTaskRepository()
    processor = DelayedNoopProcessor(processor_delay_ms / 1000)
    runtime = VideoTaskRuntime(
        queue=queue,
        repository=repository,
        processor=processor,
        worker_id=f"benchmark-worker-{suffix}",
        heartbeat_seconds=60,
    )
    tracker = RunTracker()
    malformed_count = round(task_count * malformed_rate)
    redis_peak = 0
    stop_sampling = asyncio.Event()

    async def sample_redis_memory(baseline: int) -> None:
        nonlocal redis_peak
        while not stop_sampling.is_set():
            info = await admin.info("memory")
            redis_peak = max(redis_peak, int(info["used_memory"]) - baseline)
            try:
                await asyncio.wait_for(stop_sampling.wait(), timeout=0.02)
            except TimeoutError:
                pass

    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while True:
            index = next_index
            next_index += 1
            if index >= task_count:
                return
            try:
                delivery = await queue.receive()
                outcome = await runtime.handle_delivery(delivery)
                if outcome != "completed":
                    tracker.errors.append(f"{delivery.message.task_id}:{outcome}")
                tracker.finished(_task_id(delivery.message), time.perf_counter())
            except Exception as exc:
                tracker.errors.append(f"worker:{type(exc).__name__}:{exc}")

    async def produce() -> float:
        malformed_emitted = 0
        started = time.perf_counter()
        for index in range(task_count):
            target_malformed = math.floor((index + 1) * malformed_count / task_count)
            while malformed_emitted < target_malformed:
                malformed_emitted += 1
                await admin.xadd(
                    stream,
                    {
                        "task_id": f"malformed-{suffix}-{malformed_emitted}",
                        "route_key": "mps_video",
                        "source_uri": "file:///sensitive/benchmark-source.mp4",
                    },
                )
            message = VideoTaskMessage(
                job_id=f"job-{suffix}-{index}",
                workspace_id="workspace-benchmark",
                source_file_id=f"source-{suffix}-{index}",
                generation=1,
                source_uri=f"file:///benchmark/{suffix}/{index}.mp4",
            )
            await repository.register(message)
            tracker.published(_task_id(message), time.perf_counter())
            await queue.publish(message)
        return time.perf_counter() - started

    try:
        await admin.ping()
        await queue.start()
        baseline_memory = int((await admin.info("memory"))["used_memory"])
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        sampler = asyncio.create_task(sample_redis_memory(baseline_memory))
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        producer_seconds = await produce()
        await asyncio.gather(*workers)
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        stop_sampling.set()
        await sampler
        quarantined_count = int(await admin.xlen(quarantine_stream))
        pending = await admin.xpending(stream, group)
        final_pending_count = int(pending["pending"])
        latencies = sorted(tracker.latencies_ms)
        return BenchmarkResult(
            profile=profile,
            task_count=task_count,
            malformed_count=malformed_count,
            concurrency=concurrency,
            processor_delay_ms=processor_delay_ms,
            wall_seconds=wall_seconds,
            producer_seconds=producer_seconds,
            publish_throughput_per_second=(task_count + malformed_count) / producer_seconds,
            completion_throughput_per_second=tracker.completed / wall_seconds,
            latency_p50_ms=_percentile(latencies, 50),
            latency_p95_ms=_percentile(latencies, 95),
            latency_p99_ms=_percentile(latencies, 99),
            latency_max_ms=max(latencies, default=0.0),
            peak_inflight=tracker.peak_inflight,
            average_cpu_cores=cpu_seconds / wall_seconds,
            process_peak_rss_mb=_peak_rss_mb(),
            redis_peak_delta_mb=max(redis_peak, 0) / (1024 * 1024),
            completed_count=tracker.completed,
            quarantined_count=quarantined_count,
            final_pending_count=final_pending_count,
            error_count=len(tracker.errors),
        )
    finally:
        stop_sampling.set()
        await queue.close()
        await admin.delete(stream, dlq_stream, quarantine_stream, delayed_key)
        await admin.aclose()


def _task_id(message: VideoTaskMessage) -> str:
    if message.task_id is None:  # pragma: no cover - message post-init invariant
        raise RuntimeError("benchmark message has no task id")
    return message.task_id


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_value = raw if sys.platform == "darwin" else raw * 1024
    return bytes_value / (1024 * 1024)


async def _main() -> None:
    args = _arguments()
    results: list[BenchmarkResult] = []
    for concurrency in dict.fromkeys(args.concurrency):
        result = await _run_once(
            redis_url=args.redis_url,
            profile=args.profile,
            task_count=args.task_count,
            concurrency=concurrency,
            processor_delay_ms=args.processor_delay_ms,
            malformed_rate=args.malformed_rate,
        )
        results.append(result)
        print(json.dumps(asdict(result), ensure_ascii=False))
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": os.uname().nodename,
        "scope": "real Redis + production runtime; in-memory repository; no database writes",
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(_main())
