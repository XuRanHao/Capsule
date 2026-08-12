"""Pressure-test 50 video, 100 image and 50 text durable tasks to Asset storage."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmark_image_text_asset_persistence import _select_images, _select_texts
from redis.asyncio import Redis
from sqlalchemy import func, select, text

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile
from capsule.db.processing_task_persistence import PostgresCpuProcessingTaskRepository
from capsule.db.session import Database
from capsule.db.video_tasks import PostgresVideoTaskRepository, VideoProcessingTask
from capsule.parsers.discovery import sha256_file
from capsule.pipeline.processing_task_service import (
    CpuProcessingTaskWorker,
    ProcessingTaskSubmissionService,
    _new_cpu_queue,
)
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    RedisVideoTaskQueue,
)
from capsule.pipeline.video_task_service import VideoTaskSubmissionService, VideoTaskWorker


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-manifest",
        type=Path,
        default=Path("data/benchmarks/logical-video-50-2026-08-11.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path.home() / "Downloads",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--video-count", type=int, default=50)
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--text-count", type=int, default=50)
    parser.add_argument("--video-concurrency", type=int, default=2)
    parser.add_argument("--image-concurrency", type=int, default=16)
    parser.add_argument("--text-concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=1_800)
    parser.add_argument("--workspace")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/mixed-durable-assets-200-2026-08-12.json"),
    )
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
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


def _load_videos(manifest: Path, count: int) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = list(payload["sample"]["videos"][:count])
    if len(rows) != count or any(not Path(row["path"]).is_file() for row in rows):
        raise ValueError("video manifest does not contain the requested existing samples")
    return rows


def _add_hashes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        path = Path(row["path"])
        return {**row, "sha256": sha256_file(path)}

    with ThreadPoolExecutor(max_workers=16) as executor:
        return list(executor.map(enrich, rows))


def _rss_bytes(root_pid: int) -> int:
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


async def _monitor(resources: dict[str, int], stop: asyncio.Event) -> None:
    while not stop.is_set():
        resources["peak_rss_bytes"] = max(
            resources["peak_rss_bytes"],
            await asyncio.to_thread(_rss_bytes, os.getpid()),
        )
        try:
            import torch

            resources["peak_mps_allocated_bytes"] = max(
                resources["peak_mps_allocated_bytes"],
                int(torch.mps.current_allocated_memory()),
            )
            resources["peak_mps_driver_bytes"] = max(
                resources["peak_mps_driver_bytes"],
                int(torch.mps.driver_allocated_memory()),
            )
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


async def _task_states(database: Database, workspace_id: str) -> dict[str, int]:
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


async def _database_snapshot(database: Database, workspace_id: str) -> dict[str, Any]:
    async with database.session() as session:
        tasks = list(
            await session.scalars(
                select(VideoProcessingTask)
                .join(ProcessingJob, ProcessingJob.job_id == VideoProcessingTask.parent_job_id)
                .where(ProcessingJob.workspace_id == workspace_id)
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
        stored_counts = dict(
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
                    "select count(*) from model_call_logs m join assets a "
                    "on a.asset_id=m.asset_id where a.workspace_id=:workspace"
                ),
                {"workspace": workspace_id},
            )
            or 0
        )
        downstream: dict[str, int] = {}
        for table_name in ("embedding_records", "cluster_runs", "cluster_capsules", "clusters"):
            downstream[table_name] = int(
                await session.scalar(
                    text(f"select count(*) from {table_name} where workspace_id=:workspace"),
                    {"workspace": workspace_id},
                )
                or 0
            )
        for table_name in (
            "cluster_memberships",
            "cluster_representative_assets",
            "cluster_members",
        ):
            downstream[table_name] = int(
                await session.scalar(
                    text(
                        f"select count(*) from {table_name} c join assets a "
                        "on a.asset_id=c.asset_id where a.workspace_id=:workspace"
                    ),
                    {"workspace": workspace_id},
                )
                or 0
            )

    mismatches: list[str] = []
    latency_by_kind: dict[str, list[float]] = defaultdict(list)
    for task in tasks:
        expected = int((task.result or {}).get("metadata", {}).get("asset_count", -1))
        if stored_counts.get(task.source_file_id, 0) != expected:
            mismatches.append(task.task_id)
        latency_by_kind[task.task_kind].append((task.updated_at - task.created_at).total_seconds())

    video_continuity_errors = 0
    video_assets: dict[str, list[Asset]] = defaultdict(list)
    for asset in assets:
        if asset.asset_type == "video_segment":
            video_assets[asset.source_file_id].append(asset)
    for source_assets in video_assets.values():
        ordered = sorted(source_assets, key=lambda asset: asset.source_locator["segment_index"])
        indices = [asset.source_locator["segment_index"] for asset in ordered]
        starts = [asset.source_locator["start_ms"] for asset in ordered]
        ends = [asset.source_locator["end_ms"] for asset in ordered]
        duration = int(ordered[0].file_info["source_duration_ms"])
        if not (
            indices == list(range(len(ordered)))
            and starts[0] == 0
            and ends[-1] == duration
            and all(ends[index] == starts[index + 1] for index in range(len(ordered) - 1))
        ):
            video_continuity_errors += 1

    return {
        "counts": {
            "jobs": len(jobs),
            "sources": len(sources),
            "tasks": len(tasks),
            "assets": len(assets),
        },
        "task_kinds": dict(Counter(task.task_kind for task in tasks)),
        "task_statuses": dict(Counter(task.status for task in tasks)),
        "task_attempts": dict(Counter(str(task.attempt) for task in tasks)),
        "task_routes": dict(Counter(task.route_key for task in tasks)),
        "job_statuses": dict(Counter(job.status for job in jobs)),
        "source_statuses": dict(Counter(source.processing_status for source in sources)),
        "asset_types": dict(Counter(asset.asset_type for asset in assets)),
        "asset_statuses": dict(Counter(asset.processing_status for asset in assets)),
        "task_asset_count_mismatches": mismatches,
        "video_continuity_errors": video_continuity_errors,
        "duplicate_asset_ids": len(assets) - len({asset.asset_id for asset in assets}),
        "duplicate_source_asset_keys": len(assets)
        - len({(asset.source_file_id, asset.asset_key) for asset in assets}),
        "assets_with_description": sum(bool(asset.asset_description) for asset in assets),
        "assets_with_features": sum(bool(asset.asset_features) for asset in assets),
        "model_call_logs": model_calls,
        "end_to_end_ms_by_kind": {
            kind: _distribution(values, scale=1000) for kind, values in latency_by_kind.items()
        },
        **downstream,
    }


async def _redis_route_snapshot(
    client: Redis,
    *,
    stream: str,
    group: str,
    dlq: str,
) -> dict[str, int]:
    # The consumer group must exist after a completed benchmark. Any Redis
    # error, including NOGROUP, invalidates the queue-drain evidence.
    pending = int((await client.xpending(stream, group))["pending"])
    return {
        "pending": pending,
        "delayed": int(await client.zcard(f"{stream}:delayed")),
        "dlq": int(await client.xlen(dlq)),
        "quarantine": int(await client.xlen(f"{stream}:quarantine")),
    }


async def _main() -> None:
    args = _arguments()
    videos = _load_videos(args.video_manifest, args.video_count)
    images = await asyncio.to_thread(_select_images, args.image_root, args.image_count)
    texts = await asyncio.to_thread(
        _select_texts,
        args.download_root,
        args.project_root,
        args.text_count,
    )
    videos, images, texts = await asyncio.gather(
        asyncio.to_thread(_add_hashes, videos),
        asyncio.to_thread(_add_hashes, images),
        asyncio.to_thread(_add_hashes, texts),
    )
    run_id = uuid4().hex[:10]
    workspace_id = args.workspace or f"bench_mixed_tasks_{datetime.now(UTC):%Y%m%d}_{run_id}"
    prefix = f"capsule:bench:mixed:{run_id}"
    base = get_settings()
    settings = base.model_copy(
        update={
            "ffmpeg_concurrency": args.video_concurrency,
            "video_output_mode": "logical",
            "video_source_roots": [args.image_root],
            "cpu_source_roots": [args.download_root, args.project_root],
            "cpu_image_task_concurrency": args.image_concurrency,
            "cpu_text_task_concurrency": args.text_concurrency,
            "video_task_stream": f"{prefix}:video",
            "video_task_group": f"mixed-video-{run_id}",
            "video_task_dlq_stream": f"{prefix}:video:dlq",
            "cpu_image_task_stream": f"{prefix}:image",
            "cpu_image_task_group": f"mixed-image-{run_id}",
            "cpu_image_task_dlq_stream": f"{prefix}:image:dlq",
            "cpu_text_task_stream": f"{prefix}:text",
            "cpu_text_task_group": f"mixed-text-{run_id}",
            "cpu_text_task_dlq_stream": f"{prefix}:text:dlq",
            "ark_api_key": None,
            "deepseek_api_key": None,
            "milvus_uri": "http://127.0.0.1:1",
        }
    )
    database = Database(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    resources = {
        "peak_rss_bytes": 0,
        "peak_mps_allocated_bytes": 0,
        "peak_mps_driver_bytes": 0,
    }
    stop_monitor = asyncio.Event()
    monitor = asyncio.create_task(_monitor(resources, stop_monitor))
    workers: list[asyncio.Task[None]] = []
    video_submit_queue = RedisVideoTaskQueue(
        redis_url=settings.redis_url,
        stream=settings.video_task_stream,
        group=settings.video_task_group,
        consumer=f"submit-video-{run_id}",
        dlq_stream=settings.video_task_dlq_stream,
        claim_idle_ms=settings.video_task_claim_idle_ms,
    )
    image_submit_queue = _new_cpu_queue(
        settings,
        task_kind=ProcessingTaskKind.IMAGE,
        consumer=f"submit-image-{run_id}",
    )
    text_submit_queue = _new_cpu_queue(
        settings,
        task_kind=ProcessingTaskKind.TEXT,
        consumer=f"submit-text-{run_id}",
    )
    cpu_repositories = {
        ProcessingTaskKind.IMAGE: PostgresCpuProcessingTaskRepository(
            database.session_factory,
            task_kind=ProcessingTaskKind.IMAGE,
        ),
        ProcessingTaskKind.TEXT: PostgresCpuProcessingTaskRepository(
            database.session_factory,
            task_kind=ProcessingTaskKind.TEXT,
        ),
    }
    cpu_service = ProcessingTaskSubmissionService(
        settings=settings,
        database=database,
        queues={
            ProcessingTaskKind.IMAGE: image_submit_queue,
            ProcessingTaskKind.TEXT: text_submit_queue,
        },
        task_repositories=cpu_repositories,
    )
    video_service = VideoTaskSubmissionService(
        settings=settings,
        database=database,
        task_repository=PostgresVideoTaskRepository(database.session_factory),
        queue=video_submit_queue,
    )
    submissions: dict[str, list[float]] = defaultdict(list)
    benchmark_started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    try:
        for index, row in enumerate(videos):
            started = time.perf_counter()
            await video_service.submit(
                input_path=Path(row["path"]),
                workspace_id=workspace_id,
            )
            submissions["video"].append(time.perf_counter() - started)
            if (index + 1) % 10 == 0:
                print(f"submitted video={index + 1}/{len(videos)}", flush=True)
        for index, row in enumerate(images):
            started = time.perf_counter()
            await cpu_service.submit(
                input_path=Path(row["path"]),
                workspace_id=workspace_id,
                relative_path=f"images/image_{index:03d}{row['extension']}",
            )
            submissions["image"].append(time.perf_counter() - started)
            if (index + 1) % 25 == 0:
                print(f"submitted image={index + 1}/{len(images)}", flush=True)
        for index, row in enumerate(texts):
            started = time.perf_counter()
            await cpu_service.submit(
                input_path=Path(row["path"]),
                workspace_id=workspace_id,
                relative_path=f"texts/text_{index:03d}{row['extension']}",
            )
            submissions["text"].append(time.perf_counter() - started)
            if (index + 1) % 10 == 0:
                print(f"submitted text={index + 1}/{len(texts)}", flush=True)
        submission_wall_seconds = time.perf_counter() - benchmark_started
        expected = args.video_count + args.image_count + args.text_count
        queued = await _task_states(database, workspace_id)
        if queued != {"queued": expected}:
            raise RuntimeError(f"expected all {expected} tasks queued before workers: {queued}")

        video_worker = VideoTaskWorker.from_settings(
            settings=settings,
            worker_id=f"mixed-video-{run_id}",
        )
        image_worker = CpuProcessingTaskWorker.for_kind(
            task_kind=ProcessingTaskKind.IMAGE,
            settings=settings,
            worker_id=f"mixed-image-{run_id}",
        )
        text_worker = CpuProcessingTaskWorker.for_kind(
            task_kind=ProcessingTaskKind.TEXT,
            settings=settings,
            worker_id=f"mixed-text-{run_id}",
        )
        processing_started = time.perf_counter()
        workers = [
            asyncio.create_task(video_worker.run_forever()),
            asyncio.create_task(image_worker.run_forever()),
            asyncio.create_task(text_worker.run_forever()),
        ]
        deadline = time.perf_counter() + args.timeout_seconds
        last_completed = -1
        while time.perf_counter() < deadline:
            states = await _task_states(database, workspace_id)
            completed = states.get("completed", 0)
            failed = states.get("failed", 0)
            if completed != last_completed and (completed % 20 == 0 or failed):
                print(f"completed={completed}/{expected} states={states}", flush=True)
                last_completed = completed
            if completed + failed == expected:
                break
            if any(worker.done() for worker in workers):
                for worker in workers:
                    if worker.done():
                        await worker
                raise RuntimeError("a mixed worker exited before all tasks became terminal")
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("mixed durable benchmark timed out")
        processing_wall_seconds = time.perf_counter() - processing_started
        snapshot = await _database_snapshot(database, workspace_id)
        redis_snapshot = {
            "video": await _redis_route_snapshot(
                redis,
                stream=settings.video_task_stream,
                group=settings.video_task_group,
                dlq=settings.video_task_dlq_stream,
            ),
            "image": await _redis_route_snapshot(
                redis,
                stream=settings.cpu_image_task_stream,
                group=settings.cpu_image_task_group,
                dlq=settings.cpu_image_task_dlq_stream,
            ),
            "text": await _redis_route_snapshot(
                redis,
                stream=settings.cpu_text_task_stream,
                group=settings.cpu_text_task_group,
                dlq=settings.cpu_text_task_dlq_stream,
            ),
        }
        total_wall_seconds = time.perf_counter() - benchmark_started
        cpu_finished = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = (cpu_finished.ru_utime + cpu_finished.ru_stime) - (
            cpu_started.ru_utime + cpu_started.ru_stime
        )
        passed = all(
            (
                snapshot["counts"]["jobs"] == expected,
                snapshot["counts"]["sources"] == expected,
                snapshot["counts"]["tasks"] == expected,
                snapshot["task_kinds"]
                == {"video": args.video_count, "image": args.image_count, "text": args.text_count},
                snapshot["task_routes"]
                == {
                    "mps_video": args.video_count,
                    "cpu_image": args.image_count,
                    "cpu_text": args.text_count,
                },
                snapshot["task_statuses"] == {"completed": expected},
                snapshot["task_attempts"] == {"1": expected},
                snapshot["job_statuses"] == {"completed": expected},
                snapshot["source_statuses"] == {"completed": expected},
                not snapshot["task_asset_count_mismatches"],
                snapshot["video_continuity_errors"] == 0,
                snapshot["duplicate_asset_ids"] == 0,
                snapshot["duplicate_source_asset_keys"] == 0,
                snapshot["assets_with_description"] == 0,
                snapshot["assets_with_features"] == 0,
                snapshot["model_call_logs"] == 0,
                snapshot["embedding_records"] == 0,
                snapshot["cluster_runs"] == 0,
                snapshot["cluster_capsules"] == 0,
                snapshot["clusters"] == 0,
                snapshot["cluster_memberships"] == 0,
                snapshot["cluster_representative_assets"] == 0,
                snapshot["cluster_members"] == 0,
                all(value == 0 for route in redis_snapshot.values() for value in route.values()),
            )
        )
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "workspace_id": workspace_id,
            "scope": {
                "durable_postgresql_tasks": True,
                "redis_routes": ["mps_video", "cpu_image", "cpu_text"],
                "postgresql_assets": True,
                "understanding": False,
                "external_embedding": False,
                "milvus_write": False,
                "clustering": False,
                "database_data_retained": True,
            },
            "configuration": {
                "video_count": args.video_count,
                "image_count": args.image_count,
                "text_count": args.text_count,
                "video_concurrency": args.video_concurrency,
                "image_concurrency": args.image_concurrency,
                "text_concurrency": args.text_concurrency,
            },
            "performance": {
                "submission_wall_seconds": round(submission_wall_seconds, 3),
                "processing_wall_seconds": round(processing_wall_seconds, 3),
                "total_wall_seconds": round(total_wall_seconds, 3),
                "files_per_second": round(expected / processing_wall_seconds, 3),
                "average_cpu_cores": round(cpu_seconds / total_wall_seconds, 3),
                "peak_process_tree_rss_mb": round(resources["peak_rss_bytes"] / 2**20, 3),
                "peak_mps_allocated_mb": round(resources["peak_mps_allocated_bytes"] / 2**20, 3),
                "peak_mps_driver_mb": round(resources["peak_mps_driver_bytes"] / 2**20, 3),
                "submission_ms_by_kind": {
                    kind: _distribution(values, scale=1000) for kind, values in submissions.items()
                },
            },
            "database": snapshot,
            "redis": redis_snapshot,
            "samples": {"videos": videos, "images": images, "texts": texts},
            "acceptance": {"passed": passed},
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
                    "passed": passed,
                    "performance": report["performance"],
                    "database": snapshot["counts"],
                    "task_kinds": snapshot["task_kinds"],
                    "asset_types": snapshot["asset_types"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if not passed:
            raise RuntimeError("mixed durable benchmark acceptance failed")
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await video_submit_queue.close()
        await image_submit_queue.close()
        await text_submit_queue.close()
        stop_monitor.set()
        await monitor
        await database.dispose()
        keys: list[str] = []
        for stream, dlq in (
            (settings.video_task_stream, settings.video_task_dlq_stream),
            (settings.cpu_image_task_stream, settings.cpu_image_task_dlq_stream),
            (settings.cpu_text_task_stream, settings.cpu_text_task_dlq_stream),
        ):
            keys.extend((stream, dlq, f"{stream}:quarantine", f"{stream}:delayed"))
        await redis.delete(*keys)
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
