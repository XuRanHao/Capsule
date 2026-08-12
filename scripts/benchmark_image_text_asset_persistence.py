"""Pressure-test real image/text assetization through PostgreSQL persistence.

This benchmark uses the production PipelineRunner without an enrichment
callback. It therefore persists SourceFiles and Assets but does not construct
understanding, external embedding, Milvus, or clustering services.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from sqlalchemy import func, select, text

from capsule.config import get_settings
from capsule.db.models import Asset, ProcessingJob, SourceFile
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.model_clients.tokenization import LocalTokenCounter
from capsule.pipeline.runner import PipelineRunner

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXTENSIONS = {".md", ".txt"}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "tmp",
}


@dataclass(slots=True)
class Resources:
    peak_rss_bytes: int = 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--image-count", type=int, default=100)
    parser.add_argument("--text-count", type=int, default=50)
    parser.add_argument("--image-concurrency", type=int, default=16)
    parser.add_argument("--text-concurrency", type=int, default=4)
    parser.add_argument("--workspace")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/image-text-asset-persistence-150-2026-08-12.json"),
    )
    parser.add_argument("--staging-root", type=Path, default=Path("tmp/mixed-asset-samples"))
    args = parser.parse_args()
    if min(args.image_count, args.text_count) < 1:
        parser.error("sample counts must be positive")
    if min(args.image_concurrency, args.text_concurrency) < 1:
        parser.error("concurrency must be positive")
    return args


def _even_positions(length: int, count: int) -> list[int]:
    if count > length:
        raise ValueError(f"cannot select {count} files from {length} candidates")
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _image_info(path: Path) -> dict[str, Any] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
        return {
            "path": str(path.resolve()),
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "format": image_format,
        }
    except (OSError, ValueError):
        return None


def _select_images(root: Path, count: int) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        inventory = [row for row in executor.map(_image_info, paths) if row is not None]
    by_extension: dict[str, list[dict[str, Any]]] = {}
    for row in inventory:
        by_extension.setdefault(row["extension"], []).append(row)
    for rows in by_extension.values():
        rows.sort(key=lambda row: (row["file_size_bytes"], row["path"]))

    available = Counter(row["extension"] for row in inventory)
    quotas: dict[str, int] = {}
    assigned = 0
    ordered_extensions = sorted(available, key=lambda ext: (-available[ext], ext))
    for extension in ordered_extensions:
        quota = min(available[extension], math.floor(count * available[extension] / len(inventory)))
        quotas[extension] = quota
        assigned += quota
    while assigned < count:
        for extension in ordered_extensions:
            if quotas[extension] < available[extension]:
                quotas[extension] += 1
                assigned += 1
                if assigned == count:
                    break

    selected: list[dict[str, Any]] = []
    for extension in ordered_extensions:
        rows = by_extension[extension]
        selected.extend(rows[index] for index in _even_positions(len(rows), quotas[extension]))
    return sorted(selected, key=lambda row: row["path"])


def _text_candidates(root: Path) -> list[Path]:
    rows: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        size = path.stat().st_size
        if 100 <= size <= 1_000_000:
            rows.append(path.resolve())
    return sorted(set(rows))


def _select_texts(download_root: Path, project_root: Path, count: int) -> list[dict[str, Any]]:
    downloads = _text_candidates(download_root)
    selected_paths = list(downloads[:count])
    remaining = count - len(selected_paths)
    if remaining > 0:
        project = [path for path in _text_candidates(project_root) if path not in selected_paths]
        project.sort(key=lambda path: (path.stat().st_size, str(path)))
        selected_paths.extend(project[index] for index in _even_positions(len(project), remaining))
    if len(selected_paths) != count:
        raise ValueError(f"found only {len(selected_paths)} real text files")
    return [
        {
            "path": str(path),
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "source_group": "下载" if download_root in path.parents else "项目文档",
        }
        for path in selected_paths
    ]


def _stage_samples(
    staging_root: Path,
    *,
    images: list[dict[str, Any]],
    texts: list[dict[str, Any]],
) -> tuple[Path, Path]:
    image_root = staging_root / "images"
    text_root = staging_root / "texts"
    image_root.mkdir(parents=True, exist_ok=False)
    text_root.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(images):
        (image_root / f"image_{index:03d}{row['extension']}").symlink_to(row["path"])
    for index, row in enumerate(texts):
        (text_root / f"text_{index:03d}{row['extension']}").symlink_to(row["path"])
    return image_root, text_root


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


async def _monitor_resources(resources: Resources, stop: asyncio.Event) -> None:
    while not stop.is_set():
        resources.peak_rss_bytes = max(
            resources.peak_rss_bytes,
            await asyncio.to_thread(_process_tree_rss_bytes, os.getpid()),
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


async def _database_snapshot(database: Database, workspace_id: str) -> dict[str, Any]:
    async with database.session() as session:
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
        assets_by_source = dict(
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
                    "SELECT count(*) FROM model_call_logs m JOIN assets a "
                    "ON a.asset_id=m.asset_id WHERE a.workspace_id=:workspace"
                ),
                {"workspace": workspace_id},
            )
            or 0
        )
        downstream: dict[str, int] = {}
        for table_name in ("embedding_records", "cluster_runs", "cluster_capsules", "clusters"):
            downstream[table_name] = int(
                await session.scalar(
                    text(f"SELECT count(*) FROM {table_name} WHERE workspace_id=:workspace"),
                    {"workspace": workspace_id},
                )
                or 0
            )
    return {
        "counts": {"jobs": len(jobs), "sources": len(sources), "assets": len(assets)},
        "job_statuses": dict(Counter(job.status for job in jobs)),
        "source_statuses": dict(Counter(source.processing_status for source in sources)),
        "source_extensions": dict(Counter(source.file_type for source in sources)),
        "asset_types": dict(Counter(asset.asset_type for asset in assets)),
        "asset_statuses": dict(Counter(asset.processing_status for asset in assets)),
        "sources_without_assets": sum(
            assets_by_source.get(source.source_file_id, 0) == 0 for source in sources
        ),
        "assets_with_description": sum(bool(asset.asset_description) for asset in assets),
        "assets_with_features": sum(bool(asset.asset_features) for asset in assets),
        "assets_with_derived_uri": sum(
            asset.derived_file_uri is not None or asset.preview_uri is not None for asset in assets
        ),
        "model_call_logs": model_calls,
        **downstream,
        "jobs": [
            {
                "job_id": job.job_id,
                "input_path": job.input_path,
                "status": job.status,
                "total_count": job.total_count,
                "completed_count": job.completed_count,
                "failed_count": job.failed_count,
                "retry_count": job.retry_count,
                "stage_durations_ms": job.stage_durations_ms,
            }
            for job in jobs
        ],
    }


async def _main() -> None:
    args = _arguments()
    images = await asyncio.to_thread(_select_images, args.image_root, args.image_count)
    texts = await asyncio.to_thread(
        _select_texts,
        args.download_root,
        args.project_root,
        args.text_count,
    )
    run_id = uuid4().hex[:10]
    workspace_id = args.workspace or f"bench_image_text_{datetime.now(UTC):%Y%m%d}_{run_id}"
    staging_root = (args.staging_root / run_id).resolve()
    image_root, text_root = _stage_samples(staging_root, images=images, texts=texts)

    base = get_settings()
    common_updates = {
        "video_output_mode": "logical",
        "ark_api_key": None,
        "deepseek_api_key": None,
        "milvus_uri": "http://127.0.0.1:1",
    }
    image_settings = base.model_copy(
        update={**common_updates, "file_parse_concurrency": args.image_concurrency}
    )
    text_settings = base.model_copy(
        update={**common_updates, "file_parse_concurrency": args.text_concurrency}
    )
    database = Database(base)
    repository = AssetRepository(database)
    await repository.create_workspace(name=workspace_id, workspace_id=workspace_id)
    resources = Resources()
    stop = asyncio.Event()
    monitor = asyncio.create_task(_monitor_resources(resources, stop))
    started = time.perf_counter()
    try:
        image_runner = PipelineRunner(settings=image_settings, database=database)
        text_runner = PipelineRunner(
            settings=text_settings,
            database=database,
            token_counter=LocalTokenCounter(
                text_settings.document_tokenizer_path,
                batch_size=text_settings.tokenization_batch_size,
            ),
        )

        async def run_one(kind: str, runner: PipelineRunner, root: Path) -> dict[str, Any]:
            item_started = time.perf_counter()
            result = await runner.run(root, workspace_id, on_assets_stored=None)
            return {
                "kind": kind,
                "wall_seconds": time.perf_counter() - item_started,
                **result.model_dump(mode="json"),
            }

        image_result, text_result = await asyncio.gather(
            run_one("image", image_runner, image_root),
            run_one("text", text_runner, text_root),
        )
        wall_seconds = time.perf_counter() - started
        database_snapshot = await _database_snapshot(database, workspace_id)
        expected_sources = args.image_count + args.text_count
        expected_asset_minimum = expected_sources
        passed = all(
            (
                image_result["succeeded_count"] == args.image_count,
                image_result["failed_count"] == 0,
                text_result["succeeded_count"] == args.text_count,
                text_result["failed_count"] == 0,
                database_snapshot["counts"]["jobs"] == 2,
                database_snapshot["counts"]["sources"] == expected_sources,
                database_snapshot["counts"]["assets"] >= expected_asset_minimum,
                database_snapshot["job_statuses"] == {"completed": 2},
                database_snapshot["source_statuses"] == {"completed": expected_sources},
                database_snapshot["sources_without_assets"] == 0,
                database_snapshot["asset_statuses"]
                == {"pending": database_snapshot["counts"]["assets"]},
                database_snapshot["assets_with_description"] == 0,
                database_snapshot["assets_with_features"] == 0,
                database_snapshot["assets_with_derived_uri"] == 0,
                database_snapshot["model_call_logs"] == 0,
                database_snapshot["embedding_records"] == 0,
                database_snapshot["cluster_runs"] == 0,
                database_snapshot["cluster_capsules"] == 0,
                database_snapshot["clusters"] == 0,
            )
        )
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "workspace_id": workspace_id,
            "staging_root": str(staging_root),
            "scope": {
                "postgresql_sources_jobs_assets": True,
                "durable_processing_task_runtime": False,
                "understanding": False,
                "external_embedding": False,
                "milvus_write": False,
                "clustering": False,
                "database_data_retained": True,
                "staging_symlinks_retained": True,
            },
            "configuration": {
                "image_count": args.image_count,
                "text_count": args.text_count,
                "image_concurrency": args.image_concurrency,
                "text_concurrency": args.text_concurrency,
            },
            "performance": {
                "mixed_wall_seconds": round(wall_seconds, 3),
                "files_per_second": round(expected_sources / wall_seconds, 3),
                "peak_process_tree_rss_mb": round(resources.peak_rss_bytes / 2**20, 3),
                "image_wall_seconds": round(image_result["wall_seconds"], 3),
                "text_wall_seconds": round(text_result["wall_seconds"], 3),
            },
            "results": {"image": image_result, "text": text_result},
            "database": database_snapshot,
            "samples": {"images": images, "texts": texts},
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
                    "database": database_snapshot["counts"],
                    "asset_types": database_snapshot["asset_types"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        if not passed:
            raise RuntimeError("image/text benchmark acceptance failed")
    finally:
        stop.set()
        await monitor
        await database.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
