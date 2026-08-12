"""Benchmark durable logical-video tasks through PostgreSQL Asset persistence.

The measured path is deliberately limited to:

    PostgreSQL task/source/job -> Redis delivery -> video analysis -> Asset commit

No understanding, external embedding, Milvus write, or clustering service is
constructed by this process.  MobileCLIP is still used locally by the video
segmenter; that is part of assetization rather than downstream vector indexing.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import resource
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from redis.asyncio import Redis
from sqlalchemy import func, select, text

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.db.video_asset_committer import PostgresFencedVideoAssetCommitter
from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.model_clients.mobileclip import ResidentMobileClipWorker
from capsule.parsers.discovery import discover_files
from capsule.parsers.video import VideoParser
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_processor import CapsuleVideoTaskProcessor
from capsule.pipeline.video_task_runtime import RedisVideoTaskQueue, RetryPolicy, VideoTaskRuntime
from capsule.pipeline.video_task_service import (
    VideoTaskSubmissionService,
    VideoTaskWorker,
    _video_config,
)


@dataclass(slots=True)
class RuntimeMeasurements:
    published_at: dict[str, float] = field(default_factory=dict)
    claimed_at: dict[str, float] = field(default_factory=dict)
    completed_at: dict[str, float] = field(default_factory=dict)
    asset_final_at: dict[str, float] = field(default_factory=dict)
    parser_seconds_by_path: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    commit_seconds_by_task: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    committed_assets_by_task: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    current_inflight: int = 0
    peak_inflight: int = 0
    inflight_samples: list[tuple[float, int]] = field(default_factory=list)


@dataclass(slots=True)
class ResourceMeasurements:
    peak_rss_bytes: int = 0
    peak_mps_allocated_bytes: int = 0
    peak_mps_driver_bytes: int = 0


class TimedVideoParser:
    def __init__(self, delegate: VideoParser, measurements: RuntimeMeasurements) -> None:
        self._delegate = delegate
        self._measurements = measurements

    async def assetize(self, source_file: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return await self._delegate.assetize(source_file, **kwargs)
        finally:
            self._measurements.parser_seconds_by_path[str(source_file.path)].append(
                time.perf_counter() - started
            )


class TimedCommitter:
    def __init__(
        self,
        delegate: PostgresFencedVideoAssetCommitter,
        measurements: RuntimeMeasurements,
    ) -> None:
        self._delegate = delegate
        self._measurements = measurements

    async def validate_lease_source(self, message: Any, lease: Any) -> bool:
        return cast(bool, await self._delegate.validate_lease_source(message, lease))

    async def commit_segment(
        self,
        message: Any,
        lease: Any,
        asset: Any,
        *,
        expected_asset_count: int,
    ) -> str:
        started = time.perf_counter()
        try:
            return cast(
                str,
                await self._delegate.commit_segment(
                    message,
                    lease,
                    asset,
                    expected_asset_count=expected_asset_count,
                ),
            )
        finally:
            now = time.perf_counter()
            task_id = message.task_id
            self._measurements.commit_seconds_by_task[task_id] += now - started
            self._measurements.committed_assets_by_task[task_id] += 1
            if self._measurements.committed_assets_by_task[task_id] == expected_asset_count:
                self._measurements.asset_final_at[task_id] = now


class TimedTaskRepository:
    def __init__(
        self,
        delegate: PostgresVideoTaskRepository,
        measurements: RuntimeMeasurements,
    ) -> None:
        self._delegate = delegate
        self._measurements = measurements

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def claim_attempt(self, message: Any, **kwargs: Any) -> Any:
        lease = await self._delegate.claim_attempt(message, **kwargs)
        if lease is not None:
            now = time.perf_counter()
            self._measurements.claimed_at[message.task_id] = now
            self._measurements.current_inflight += 1
            self._measurements.peak_inflight = max(
                self._measurements.peak_inflight,
                self._measurements.current_inflight,
            )
            self._measurements.inflight_samples.append((now, self._measurements.current_inflight))
        return lease

    async def complete(self, lease: Any, result: Any) -> bool:
        completed = await self._delegate.complete(lease, result)
        now = time.perf_counter()
        if lease.task_id in self._measurements.claimed_at:
            self._measurements.completed_at[lease.task_id] = now
            self._measurements.current_inflight = max(
                0,
                self._measurements.current_inflight - 1,
            )
            self._measurements.inflight_samples.append((now, self._measurements.current_inflight))
        return cast(bool, completed)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmarks/logical-video-50-2026-08-11.json"),
    )
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/video-asset-persistence-50-c2-2026-08-12.json"),
    )
    parser.add_argument("--workspace")
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    return args


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float], *, scale: float = 1.0) -> dict[str, float]:
    scaled = [value * scale for value in values]
    return {
        "average": round(sum(scaled) / len(scaled), 3) if scaled else 0.0,
        "p50": round(_percentile(scaled, 0.50), 3),
        "p95": round(_percentile(scaled, 0.95), 3),
        "p99": round(_percentile(scaled, 0.99), 3),
        "max": round(max(scaled), 3) if scaled else 0.0,
    }


def _resource_cpu_seconds() -> float:
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return self_usage.ru_utime + self_usage.ru_stime + child_usage.ru_utime + child_usage.ru_stime


def _process_tree_rss_bytes(root_pid: int) -> int:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=False,
        capture_output=True,
        text=True,
    )
    rows: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        try:
            pid_text, parent_text, rss_text = line.split()
            rows[int(pid_text)] = (int(parent_text), int(rss_text))
        except ValueError:
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in rows.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows.get(pid, (0, 0))[1] for pid in descendants) * 1024


async def _monitor_resources(
    measurements: ResourceMeasurements,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        measurements.peak_rss_bytes = max(
            measurements.peak_rss_bytes,
            await asyncio.to_thread(_process_tree_rss_bytes, os.getpid()),
        )
        try:
            import torch

            measurements.peak_mps_allocated_bytes = max(
                measurements.peak_mps_allocated_bytes,
                int(torch.mps.current_allocated_memory()),
            )
            measurements.peak_mps_driver_bytes = max(
                measurements.peak_mps_driver_bytes,
                int(torch.mps.driver_allocated_memory()),
            )
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


def _load_samples(manifest: Path, count: int) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = list(payload["sample"]["videos"][:count])
    if len(samples) != count:
        raise ValueError(f"manifest contains {len(samples)} samples, expected {count}")
    paths = [Path(row["path"]) for row in samples]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark videos: {missing[:3]}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("benchmark workspace requires unique video basenames")
    if any(path.suffix.lower() not in {".mp4", ".mov"} for path in paths):
        raise ValueError("durable video task benchmark accepts only MP4/MOV samples")
    return samples


async def _database_snapshot(database: Database, workspace_id: str) -> dict[str, Any]:
    async with database.session() as session:
        tasks = list(
            await session.scalars(
                select(VideoProcessingTask)
                .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
                .where(ProcessingJob.workspace_id == workspace_id)
                .order_by(VideoProcessingTask.created_at)
            )
        )
        jobs = list(
            await session.scalars(
                select(ProcessingJob).where(ProcessingJob.workspace_id == workspace_id)
            )
        )
        sources = list(
            await session.scalars(select(SourceFile).where(SourceFile.workspace_id == workspace_id))
        )
        assets = list(
            await session.scalars(select(Asset).where(Asset.workspace_id == workspace_id))
        )
        asset_counts = dict(
            list(
                await session.execute(
                    select(Asset.source_file_id, func.count(Asset.asset_id))
                    .where(Asset.workspace_id == workspace_id)
                    .group_by(Asset.source_file_id)
                )
            )
        )
        model_calls = int(
            await session.scalar(
                text(
                    "SELECT count(*) FROM model_call_logs m "
                    "JOIN assets a ON a.asset_id=m.asset_id WHERE a.workspace_id=:workspace"
                ),
                {"workspace": workspace_id},
            )
            or 0
        )
        table_counts: dict[str, int] = {}
        for table_name, condition in (
            ("embedding_records", "workspace_id=:workspace"),
            ("cluster_runs", "workspace_id=:workspace"),
            ("cluster_capsules", "workspace_id=:workspace"),
            ("clusters", "workspace_id=:workspace"),
        ):
            table_counts[table_name] = int(
                await session.scalar(
                    text(f"SELECT count(*) FROM {table_name} WHERE {condition}"),
                    {"workspace": workspace_id},
                )
                or 0
            )
        for table_name in (
            "cluster_memberships",
            "cluster_representative_assets",
            "cluster_members",
        ):
            table_counts[table_name] = int(
                await session.scalar(
                    text(
                        f"SELECT count(*) FROM {table_name} c JOIN assets a "
                        "ON a.asset_id=c.asset_id WHERE a.workspace_id=:workspace"
                    ),
                    {"workspace": workspace_id},
                )
                or 0
            )

    result_asset_counts = {
        task.source_file_id: int((task.result or {}).get("metadata", {}).get("asset_count", -1))
        for task in tasks
    }
    mismatches = [
        source_id
        for source_id, expected in result_asset_counts.items()
        if asset_counts.get(source_id, 0) != expected
    ]
    logical_assets = [
        asset
        for asset in assets
        if asset.asset_type == "video_segment"
        and asset.file_info.get("video_output_mode") == "logical"
        and asset.derived_file_uri is None
        and asset.preview_uri is None
    ]
    return {
        "counts": {
            "jobs": len(jobs),
            "sources": len(sources),
            "tasks": len(tasks),
            "assets": len(assets),
        },
        "task_statuses": dict(Counter(task.status for task in tasks)),
        "task_stages": dict(Counter(task.stage for task in tasks)),
        "task_attempts": dict(Counter(str(task.attempt) for task in tasks)),
        "job_statuses": dict(Counter(job.status for job in jobs)),
        "source_statuses": dict(Counter(source.processing_status for source in sources)),
        "asset_statuses": dict(Counter(asset.processing_status for asset in assets)),
        "logical_asset_count": len(logical_assets),
        "assets_with_derived_uri": sum(
            asset.derived_file_uri is not None or asset.preview_uri is not None for asset in assets
        ),
        "assets_with_description": sum(bool(asset.asset_description) for asset in assets),
        "assets_with_features": sum(bool(asset.asset_features) for asset in assets),
        "result_asset_count_mismatches": mismatches,
        "parent_jobs_exactly_once": sum(
            job.total_count == 1
            and job.completed_count == 1
            and job.failed_count == 0
            and job.retry_count == 0
            for job in jobs
        ),
        "tasks_with_parent_accounting": sum(task.parent_accounted_at is not None for task in tasks),
        "model_call_logs": model_calls,
        **table_counts,
        "task_rows": [
            {
                "task_id": task.task_id,
                "source_file_id": task.source_file_id,
                "status": task.status,
                "stage": task.stage,
                "attempt": task.attempt,
                "result_asset_count": result_asset_counts.get(task.source_file_id),
                "stored_asset_count": asset_counts.get(task.source_file_id, 0),
                "error": task.error_message,
            }
            for task in tasks
        ],
    }


async def _terminal_counts(database: Database, workspace_id: str) -> dict[str, int]:
    async with database.session() as session:
        rows = list(
            await session.execute(
                select(VideoProcessingTask.status, func.count(VideoProcessingTask.task_id))
                .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
                .where(ProcessingJob.workspace_id == workspace_id)
                .group_by(VideoProcessingTask.status)
            )
        )
    return {status: int(count) for status, count in rows}


async def _redis_snapshot(
    client: Redis,
    *,
    stream: str,
    group: str,
    keys: list[str],
) -> dict[str, int]:
    try:
        pending = int((await client.xpending(stream, group))["pending"])
    except Exception:
        pending = 0
    result = {"pending": pending}
    for key in keys:
        key_type = await client.type(key)
        if key_type == "stream":
            result[key] = int(await client.xlen(key))
        elif key_type == "zset":
            result[key] = int(await client.zcard(key))
        else:
            result[key] = 0
    return result


async def _main() -> None:
    args = _arguments()
    samples = _load_samples(args.manifest, args.sample_count)
    run_id = uuid4().hex[:10]
    workspace_id = args.workspace or f"bench_asset_{datetime.now(UTC):%Y%m%d}_{run_id}"
    stream = f"capsule:bench:video-assets:{run_id}"
    group = f"capsule-bench-video-assets-{run_id}"
    dlq = f"{stream}:dlq"
    quarantine = f"{stream}:quarantine"
    delayed = f"{stream}:delayed"
    source_roots = sorted({Path(row["path"]).parent for row in samples})

    base = get_settings()
    settings = base.model_copy(
        update={
            "ffmpeg_concurrency": args.concurrency,
            "video_output_mode": "logical",
            "video_source_roots": source_roots,
            "video_task_stream": stream,
            "video_task_group": group,
            "video_task_dlq_stream": dlq,
            # Fail fast if a downstream service is accidentally introduced.
            "ark_api_key": None,
            "deepseek_api_key": None,
            "milvus_uri": "http://127.0.0.1:1",
        }
    )
    database = Database(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    measurements = RuntimeMeasurements()
    resources = ResourceMeasurements()
    stop_monitor = asyncio.Event()
    monitor = asyncio.create_task(_monitor_resources(resources, stop_monitor))
    worker_task: asyncio.Task[None] | None = None
    submit_queue: RedisVideoTaskQueue | None = None
    worker_queue: RedisVideoTaskQueue | None = None
    cpu_started = _resource_cpu_seconds()
    run_started = time.perf_counter()
    warmup_seconds = 0.0
    submissions: list[dict[str, Any]] = []

    parser = VideoParser(
        concurrency=args.concurrency,
        config=_video_config(settings),
        embedder=ResidentMobileClipWorker(
            model_path=Path(settings.mobileclip_model_path),
            batch_size=settings.mobileclip_batch_size,
        ),
    )
    try:
        if not args.skip_warmup:
            warmup_started = time.perf_counter()
            warm_source = discover_files(Path(samples[0]["path"]))[0]
            warm_drafts = await parser.assetize(warm_source)
            warmup_seconds = time.perf_counter() - warmup_started
            del warm_drafts
            gc.collect()

        task_repository_delegate = PostgresVideoTaskRepository(
            database.session_factory,
            lease_seconds=settings.video_task_lease_seconds,
            progress_timeout_seconds=settings.video_task_progress_timeout_seconds,
            hard_timeout_seconds=settings.video_task_hard_timeout_seconds,
            redispatch_seconds=settings.video_task_redispatch_seconds,
            max_attempts=settings.video_task_max_attempts,
        )
        task_repository = TimedTaskRepository(task_repository_delegate, measurements)
        submit_queue = RedisVideoTaskQueue(
            redis_url=settings.redis_url,
            stream=stream,
            group=group,
            consumer=f"submit-{run_id}",
            dlq_stream=dlq,
            quarantine_stream=quarantine,
            delayed_key=delayed,
            claim_idle_ms=settings.video_task_claim_idle_ms,
        )
        worker_queue = RedisVideoTaskQueue(
            redis_url=settings.redis_url,
            stream=stream,
            group=group,
            consumer=f"worker-{run_id}",
            dlq_stream=dlq,
            quarantine_stream=quarantine,
            delayed_key=delayed,
            claim_idle_ms=settings.video_task_claim_idle_ms,
        )
        submission_service = VideoTaskSubmissionService(
            settings=settings,
            database=database,
            source_repository=AssetRepository(database),
            task_repository=task_repository,
            queue=submit_queue,
        )

        submit_batch_started = time.perf_counter()
        for index, sample in enumerate(samples, start=1):
            submit_started = time.perf_counter()
            submitted = await submission_service.submit(
                input_path=Path(sample["path"]),
                workspace_id=workspace_id,
            )
            submit_finished = time.perf_counter()
            if submitted.task_id is None or submitted.already_processed:
                raise RuntimeError(f"sample unexpectedly reused: {sample['path']}")
            measurements.published_at[submitted.task_id] = submit_finished
            submissions.append(
                {
                    "index": index,
                    "path": sample["path"],
                    "duration_seconds": sample["duration_seconds"],
                    "duration_band": sample["duration_band"],
                    "job_id": submitted.job_id,
                    "source_file_id": submitted.source_file_id,
                    "task_id": submitted.task_id,
                    "submission_seconds": submit_finished - submit_started,
                }
            )
            if index % 10 == 0:
                print(f"submitted={index}/{len(samples)}", flush=True)
        submit_batch_seconds = time.perf_counter() - submit_batch_started

        queued_snapshot = await _terminal_counts(database, workspace_id)
        if queued_snapshot != {"queued": len(samples)}:
            raise RuntimeError(f"expected all tasks queued before worker start: {queued_snapshot}")

        timed_parser = TimedVideoParser(parser, measurements)
        timed_committer = TimedCommitter(
            PostgresFencedVideoAssetCommitter(database),
            measurements,
        )
        processor = CapsuleVideoTaskProcessor(
            parser=timed_parser,
            asset_factory=AssetFactory(),
            media_writer=None,
            committer=timed_committer,
            output_mode="logical",
        )
        runtime = VideoTaskRuntime(
            queue=worker_queue,
            repository=task_repository,
            processor=processor,
            worker_id=f"bench-worker-{run_id}",
            retry_policy=RetryPolicy(
                max_attempts=settings.video_task_max_attempts,
                retry_delays_seconds=tuple(settings.video_task_retry_delays_seconds),
            ),
            heartbeat_seconds=settings.video_task_heartbeat_seconds,
            expected_route_key="mps_video",
        )
        worker = VideoTaskWorker(
            runtime=runtime,
            queue=worker_queue,
            concurrency=args.concurrency,
        )
        processing_started = time.perf_counter()
        worker_task = asyncio.create_task(worker.run_forever())
        deadline = time.perf_counter() + args.timeout_seconds
        last_completed = -1
        while time.perf_counter() < deadline:
            states = await _terminal_counts(database, workspace_id)
            completed = states.get("completed", 0)
            failed = states.get("failed", 0)
            if completed != last_completed and (completed % 5 == 0 or failed):
                print(f"completed={completed}/{len(samples)} states={states}", flush=True)
                last_completed = completed
            if completed + failed == len(samples):
                break
            if worker_task.done():
                await worker_task
                raise RuntimeError("video worker exited before every task became terminal")
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError(f"benchmark exceeded {args.timeout_seconds} seconds")
        processing_wall_seconds = time.perf_counter() - processing_started

        final_db = await _database_snapshot(database, workspace_id)
        redis_final = await _redis_snapshot(
            redis,
            stream=stream,
            group=group,
            keys=[delayed, dlq, quarantine],
        )
        successful = final_db["task_statuses"].get("completed", 0)
        total_duration = sum(float(row["duration_seconds"]) for row in samples)

        queue_waits: list[float] = []
        processing_times: list[float] = []
        asset_finalize_times: list[float] = []
        transport_finalize_times: list[float] = []
        end_to_end_times: list[float] = []
        parser_times: list[float] = []
        commit_times: list[float] = []
        per_task: list[dict[str, Any]] = []
        for row in submissions:
            task_id = row["task_id"]
            path = str(row["path"])
            published = measurements.published_at[task_id]
            claimed = measurements.claimed_at.get(task_id)
            task_completed_at = measurements.completed_at.get(task_id)
            asset_final = measurements.asset_final_at.get(task_id)
            parser_seconds = sum(measurements.parser_seconds_by_path.get(path, []))
            commit_seconds = measurements.commit_seconds_by_task.get(task_id, 0.0)
            if claimed is not None:
                queue_waits.append(claimed - published)
            if claimed is not None and task_completed_at is not None:
                processing_times.append(task_completed_at - claimed)
            if claimed is not None and asset_final is not None:
                asset_finalize_times.append(asset_final - claimed)
            if asset_final is not None and task_completed_at is not None:
                transport_finalize_times.append(task_completed_at - asset_final)
            if task_completed_at is not None:
                end_to_end_times.append(task_completed_at - published)
            parser_times.append(parser_seconds)
            commit_times.append(commit_seconds)
            per_task.append(
                {
                    **row,
                    "queue_wait_seconds": None if claimed is None else claimed - published,
                    "processing_seconds": (
                        None
                        if claimed is None or task_completed_at is None
                        else task_completed_at - claimed
                    ),
                    "asset_finalize_seconds": (
                        None if claimed is None or asset_final is None else asset_final - claimed
                    ),
                    "transport_finalize_seconds": (
                        None
                        if asset_final is None or task_completed_at is None
                        else task_completed_at - asset_final
                    ),
                    "end_to_end_seconds": (
                        None if task_completed_at is None else task_completed_at - published
                    ),
                    "parser_seconds": parser_seconds,
                    "asset_commit_seconds": commit_seconds,
                    "committed_asset_count": measurements.committed_assets_by_task.get(task_id, 0),
                }
            )

        cpu_seconds = _resource_cpu_seconds() - cpu_started
        measured_wall = time.perf_counter() - run_started
        report: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "workspace_id": workspace_id,
            "scope": {
                "postgresql_tasks_sources_jobs": True,
                "postgresql_assets": True,
                "video_output_mode": "logical",
                "understanding": False,
                "external_embedding": False,
                "milvus_write": False,
                "clustering": False,
                "persistent_segment_files": False,
                "persistent_keyframe_files": False,
                "local_mobileclip_segmentation": True,
                "database_data_retained": True,
            },
            "configuration": {
                "sample_count": len(samples),
                "worker_concurrency": args.concurrency,
                "parser_ffmpeg_concurrency": settings.ffmpeg_concurrency,
                "warmup_seconds": round(warmup_seconds, 3),
                "stream": stream,
                "group": group,
                "manifest": str(args.manifest.resolve()),
            },
            "sample": {
                "video_count": len(samples),
                "total_duration_seconds": round(total_duration, 3),
                "duration_bands": dict(Counter(row["duration_band"] for row in samples)),
            },
            "performance": {
                "submission_batch_seconds": round(submit_batch_seconds, 3),
                "processing_wall_seconds": round(processing_wall_seconds, 3),
                "total_measured_wall_seconds": round(measured_wall, 3),
                "videos_per_wall_minute": round(successful / processing_wall_seconds * 60, 3),
                "source_minutes_per_wall_minute": round(
                    total_duration / processing_wall_seconds, 3
                ),
                "aggregate_realtime_factor": round(processing_wall_seconds / total_duration, 4),
                "average_cpu_cores": round(cpu_seconds / measured_wall, 3),
                "peak_process_tree_rss_mb": round(resources.peak_rss_bytes / 2**20, 3),
                "peak_mps_allocated_mb": round(resources.peak_mps_allocated_bytes / 2**20, 3),
                "peak_mps_driver_mb": round(resources.peak_mps_driver_bytes / 2**20, 3),
                "peak_inflight": measurements.peak_inflight,
                "submission_ms": _distribution(
                    [row["submission_seconds"] for row in submissions], scale=1000
                ),
                "queue_wait_ms": _distribution(queue_waits, scale=1000),
                "parser_ms": _distribution(parser_times, scale=1000),
                "asset_commit_ms": _distribution(commit_times, scale=1000),
                "asset_finalize_ms": _distribution(asset_finalize_times, scale=1000),
                "transport_finalize_ms": _distribution(transport_finalize_times, scale=1000),
                "processing_ms": _distribution(processing_times, scale=1000),
                "end_to_end_ms": _distribution(end_to_end_times, scale=1000),
            },
            "database": final_db,
            "redis": redis_final,
            "per_task": per_task,
        }
        report["acceptance"] = {
            "passed": all(
                (
                    final_db["counts"]["jobs"] == len(samples),
                    final_db["counts"]["sources"] == len(samples),
                    final_db["counts"]["tasks"] == len(samples),
                    final_db["task_statuses"] == {"completed": len(samples)},
                    final_db["task_stages"] == {"completed": len(samples)},
                    final_db["task_attempts"] == {"1": len(samples)},
                    final_db["job_statuses"] == {"completed": len(samples)},
                    final_db["source_statuses"] == {"completed": len(samples)},
                    final_db["logical_asset_count"] == final_db["counts"]["assets"],
                    final_db["asset_statuses"] == {"pending": final_db["counts"]["assets"]},
                    final_db["assets_with_derived_uri"] == 0,
                    final_db["assets_with_description"] == 0,
                    final_db["assets_with_features"] == 0,
                    not final_db["result_asset_count_mismatches"],
                    final_db["parent_jobs_exactly_once"] == len(samples),
                    final_db["tasks_with_parent_accounting"] == len(samples),
                    final_db["model_call_logs"] == 0,
                    final_db["embedding_records"] == 0,
                    final_db["cluster_runs"] == 0,
                    final_db["cluster_capsules"] == 0,
                    final_db["clusters"] == 0,
                    final_db["cluster_memberships"] == 0,
                    final_db["cluster_representative_assets"] == 0,
                    final_db["cluster_members"] == 0,
                    redis_final["pending"] == 0,
                    redis_final[delayed] == 0,
                    redis_final[dlq] == 0,
                    redis_final[quarantine] == 0,
                    measurements.peak_inflight == args.concurrency,
                    measurements.peak_inflight <= args.concurrency,
                )
            )
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "workspace_id": workspace_id,
                    "passed": report["acceptance"]["passed"],
                    "performance": report["performance"],
                    "database_counts": final_db["counts"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if not report["acceptance"]["passed"]:
            raise RuntimeError("benchmark acceptance failed; inspect output JSON")
    finally:
        if worker_task is not None:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        if worker_queue is not None:
            await worker_queue.close()
        if submit_queue is not None:
            await submit_queue.close()
        stop_monitor.set()
        await monitor
        await database.dispose()
        await redis.delete(stream, delayed, dlq, quarantine)
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
