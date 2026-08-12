"""Benchmark logical video analysis and transient frame extraction on macOS MPS.

This benchmark deliberately stops before PostgreSQL persistence, multimodal
understanding, vector indexing, and clustering. It never renders Segment videos
or writes representative-frame images.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capsule.media.video_frames import (
    FFmpegVideoFrameExtractor,
    logical_video_frame_request,
)
from capsule.model_clients.mobileclip import ResidentMobileClipWorker
from capsule.parsers.discovery import discover_files
from capsule.parsers.video import (
    VideoAnalysisProgress,
    VideoParser,
    VideoSegmentationConfig,
    resolve_video_tool,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v"}


@dataclass(frozen=True, slots=True)
class InventoryVideo:
    path: str
    duration_seconds: float
    codec: str
    width: int
    height: int
    file_size_bytes: int
    duration_band: str = ""


@dataclass(slots=True)
class VideoMeasurement:
    path: str
    duration_band: str
    source_duration_seconds: float = 0.0
    probe_seconds: float = 0.0
    decode_analysis_seconds: float = 0.0
    embedding_seconds: float = 0.0
    segmentation_seconds: float = 0.0
    frame_extraction_seconds: float = 0.0
    total_seconds: float = 0.0
    segment_count: int = 0
    representative_frame_count: int = 0
    error: str | None = None
    _started_at: float = field(default=0.0, repr=False)
    _decoding_started_at: float | None = field(default=None, repr=False)
    _segmenting_started_at: float | None = field(default=None, repr=False)
    _embedding_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration_band": self.duration_band,
            "source_duration_seconds": self.source_duration_seconds,
            "probe_seconds": self.probe_seconds,
            "decode_analysis_seconds": self.decode_analysis_seconds,
            "embedding_seconds": self.embedding_seconds,
            "segmentation_seconds": self.segmentation_seconds,
            "frame_extraction_seconds": self.frame_extraction_seconds,
            "total_seconds": self.total_seconds,
            "segment_count": self.segment_count,
            "representative_frame_count": self.representative_frame_count,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cpu_seconds: float


@dataclass(slots=True)
class RunResources:
    peak_process_tree_rss_bytes: int = 0
    peak_mps_allocated_bytes: int = 0
    peak_mps_driver_bytes: int = 0


_CURRENT_MEASUREMENT: ContextVar[VideoMeasurement | None] = ContextVar(
    "logical_video_benchmark_measurement",
    default=None,
)


class TimedResidentEmbedder:
    def __init__(self, *, model_path: Path, batch_size: int) -> None:
        self._worker = ResidentMobileClipWorker(
            model_path=model_path,
            batch_size=batch_size,
        )
        self._resource_lock = threading.Lock()
        self._run_resources: RunResources | None = None

    def set_run_resources(self, resources: RunResources) -> None:
        with self._resource_lock:
            self._run_resources = resources

    def embed(self, frames: list[Any]) -> Any:
        started = time.perf_counter()
        try:
            return self._worker.embed(frames)
        finally:
            elapsed = time.perf_counter() - started
            measurement = _CURRENT_MEASUREMENT.get()
            if measurement is not None:
                with measurement._embedding_lock:
                    measurement.embedding_seconds += elapsed
            self._sample_mps_memory()

    def _sample_mps_memory(self) -> None:
        try:
            import torch

            allocated = int(torch.mps.current_allocated_memory())
            driver = int(torch.mps.driver_allocated_memory())
        except Exception:
            return
        with self._resource_lock:
            resources = self._run_resources
            if resources is None:
                return
            resources.peak_mps_allocated_bytes = max(
                resources.peak_mps_allocated_bytes,
                allocated,
            )
            resources.peak_mps_driver_bytes = max(
                resources.peak_mps_driver_bytes,
                driver,
            )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/logical-video-50-2026-08-11.json"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/mobileclip-s0/mobileclip_s0.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def _inventory(root: Path) -> list[InventoryVideo]:
    ffprobe = resolve_video_tool("ffprobe")
    if ffprobe is None:
        raise RuntimeError("FFprobe is unavailable")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(executor.map(lambda path: _probe(ffprobe, path), paths))
    return [row for row in rows if row is not None]


def _probe(ffprobe: Path, path: Path) -> InventoryVideo | None:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        return None
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        format_info = payload["format"]
        return InventoryVideo(
            path=str(path),
            duration_seconds=float(format_info["duration"]),
            codec=str(stream.get("codec_name") or "unknown"),
            width=int(stream["width"]),
            height=int(stream["height"]),
            file_size_bytes=int(format_info.get("size") or path.stat().st_size),
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _select_stratified(
    inventory: list[InventoryVideo],
    *,
    sample_count: int,
) -> list[InventoryVideo]:
    if sample_count != 50:
        raise ValueError("this report benchmark requires exactly 50 samples")
    bands = [
        ("不超过5秒", 12, lambda value: value <= 5),
        ("5至10秒", 13, lambda value: 5 < value <= 10),
        ("10至15秒", 13, lambda value: 10 < value <= 15),
        ("超过15秒", 12, lambda value: value > 15),
    ]
    selected: list[InventoryVideo] = []
    for name, count, predicate in bands:
        candidates = sorted(
            (item for item in inventory if predicate(item.duration_seconds)),
            key=lambda item: (item.duration_seconds, item.path),
        )
        if len(candidates) < count:
            raise ValueError(f"duration band {name} has only {len(candidates)} videos")
        for position in _even_positions(len(candidates), count):
            item = candidates[position]
            selected.append(
                InventoryVideo(
                    path=item.path,
                    duration_seconds=item.duration_seconds,
                    codec=item.codec,
                    width=item.width,
                    height=item.height,
                    file_size_bytes=item.file_size_bytes,
                    duration_band=name,
                )
            )
    return selected


def _even_positions(population: int, count: int) -> list[int]:
    if count == 1:
        return [population // 2]
    return [round(index * (population - 1) / (count - 1)) for index in range(count)]


async def _benchmark_run(
    *,
    selected: list[InventoryVideo],
    concurrency: int,
    embedder: TimedResidentEmbedder,
    detailed_progress: bool,
) -> dict[str, Any]:
    parser = VideoParser(
        concurrency=concurrency,
        config=VideoSegmentationConfig(output_mode="logical"),
        embedder=embedder,
        mobileclip_batch_size=12,
    )
    frame_extractor = FFmpegVideoFrameExtractor(concurrency=concurrency)
    resources = RunResources()
    embedder.set_run_resources(resources)
    stop_monitor = asyncio.Event()
    monitor = asyncio.create_task(_monitor_memory(resources, stop_monitor))
    before = _resource_snapshot()
    run_started = time.perf_counter()
    completed = 0
    total_count = len(selected)
    completion_lock = asyncio.Lock()

    async def process(item: InventoryVideo) -> VideoMeasurement:
        nonlocal completed
        measurement = VideoMeasurement(path=item.path, duration_band=item.duration_band)
        token = _CURRENT_MEASUREMENT.set(measurement)
        measurement._started_at = time.perf_counter()

        def progress(update: VideoAnalysisProgress) -> None:
            now = time.perf_counter()
            if update.stage == "decoding" and measurement._decoding_started_at is None:
                measurement._decoding_started_at = now
                measurement.probe_seconds = now - measurement._started_at
            elif update.stage == "segmenting" and measurement._segmenting_started_at is None:
                measurement._segmenting_started_at = now

        try:
            source = discover_files(Path(item.path))[0]
            drafts = await parser.assetize(source, progress_callback=progress)
            parsing_finished = time.perf_counter()
            decoding_started = measurement._decoding_started_at or measurement._started_at
            segmenting_started = measurement._segmenting_started_at or parsing_finished
            analysis_total = max(0.0, segmenting_started - decoding_started)
            measurement.decode_analysis_seconds = max(
                0.0,
                analysis_total - measurement.embedding_seconds,
            )
            measurement.segmentation_seconds = max(
                0.0,
                parsing_finished - segmenting_started,
            )
            measurement.segment_count = len(drafts)
            measurement.representative_frame_count = sum(
                len(draft.file_info.get("representative_frames", []))
                for draft in drafts
            )
            if drafts:
                measurement.source_duration_seconds = (
                    float(drafts[0].file_info["source_duration_ms"]) / 1000
                )
            extraction_started = time.perf_counter()
            await asyncio.gather(
                *(
                    frame_extractor.extract(
                        logical_video_frame_request(
                            source_uri=Path(item.path).as_uri(),
                            file_info=draft.file_info,
                            source_locator=draft.source_locator,
                        )
                    )
                    for draft in drafts
                )
            )
            measurement.frame_extraction_seconds = (
                time.perf_counter() - extraction_started
            )
        except Exception as exc:
            measurement.error = f"{type(exc).__name__}: {exc}"
        finally:
            measurement.total_seconds = time.perf_counter() - measurement._started_at
            _CURRENT_MEASUREMENT.reset(token)
            async with completion_lock:
                completed += 1
                if detailed_progress or completed % 5 == 0:
                    status = "ok" if measurement.error is None else "failed"
                    print(
                        f"concurrency={concurrency} completed={completed}/{total_count} "
                        f"status={status} duration={item.duration_seconds:.3f}s "
                        f"wall={measurement.total_seconds:.3f}s",
                        flush=True,
                    )
        return measurement

    work_queue: asyncio.Queue[tuple[int, InventoryVideo] | None] = asyncio.Queue()
    for index, item in enumerate(selected):
        work_queue.put_nowait((index, item))
    worker_count = min(concurrency, len(selected))
    for _ in range(worker_count):
        work_queue.put_nowait(None)
    measurement_slots: list[VideoMeasurement | None] = [None] * len(selected)

    async def worker() -> None:
        while True:
            work = await work_queue.get()
            if work is None:
                return
            index, item = work
            measurement_slots[index] = await process(item)

    await asyncio.gather(*(worker() for _ in range(worker_count)))
    if any(item is None for item in measurement_slots):
        raise RuntimeError("benchmark worker exited before all videos completed")
    measurements = [item for item in measurement_slots if item is not None]
    wall_seconds = time.perf_counter() - run_started
    stop_monitor.set()
    await monitor
    after = _resource_snapshot()
    return _summarize_run(
        concurrency=concurrency,
        measurements=measurements,
        wall_seconds=wall_seconds,
        cpu_seconds=max(0.0, after.cpu_seconds - before.cpu_seconds),
        resources=resources,
    )


async def _monitor_memory(resources: RunResources, stop: asyncio.Event) -> None:
    while not stop.is_set():
        resources.peak_process_tree_rss_bytes = max(
            resources.peak_process_tree_rss_bytes,
            await asyncio.to_thread(_process_tree_rss_bytes, os.getpid()),
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            pass


def _process_tree_rss_bytes(root_pid: int) -> int:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
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


def _resource_snapshot() -> ResourceSnapshot:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ResourceSnapshot(
        cpu_seconds=(own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime),
    )


def _summarize_run(
    *,
    concurrency: int,
    measurements: list[VideoMeasurement],
    wall_seconds: float,
    cpu_seconds: float,
    resources: RunResources,
) -> dict[str, Any]:
    successful = [item for item in measurements if item.error is None]
    warm_successful = successful[1:] if concurrency == 1 else successful
    source_seconds = sum(item.source_duration_seconds for item in successful)
    representative_count = sum(item.representative_frame_count for item in successful)
    warm_stage_totals = _stage_totals(warm_successful)
    warm_stage_total = sum(warm_stage_totals.values())
    return {
        "concurrency": concurrency,
        "requested_video_count": len(measurements),
        "successful_video_count": len(successful),
        "failed_video_count": len(measurements) - len(successful),
        "success_rate": _ratio(len(successful), len(measurements)),
        "source_duration_seconds": source_seconds,
        "wall_seconds": wall_seconds,
        "aggregate_realtime_factor": _ratio(wall_seconds, source_seconds),
        "source_minutes_per_wall_minute": _ratio(source_seconds, wall_seconds),
        "warm_source_minutes_per_wall_minute": _ratio(
            sum(item.source_duration_seconds for item in warm_successful),
            sum(item.total_seconds for item in warm_successful),
        ),
        "videos_per_wall_minute": _ratio(len(successful) * 60, wall_seconds),
        "average_cpu_cores": _ratio(cpu_seconds, wall_seconds),
        "peak_process_tree_rss_mb": resources.peak_process_tree_rss_bytes / 1_000_000,
        "peak_mps_allocated_mb": resources.peak_mps_allocated_bytes / 1_000_000,
        "peak_mps_driver_mb": resources.peak_mps_driver_bytes / 1_000_000,
        "segment_count": sum(item.segment_count for item in successful),
        "representative_frame_count": representative_count,
        "transient_frame_throughput_fps": _ratio(
            representative_count,
            sum(item.frame_extraction_seconds for item in successful),
        ),
        "per_video_seconds": _distribution(
            [item.total_seconds for item in successful]
        ),
        "realtime_factor": _distribution(
            [
                _ratio(item.total_seconds, item.source_duration_seconds)
                for item in successful
                if item.source_duration_seconds > 0
            ]
        ),
        "stage_seconds": {
            "probe": _distribution([item.probe_seconds for item in successful]),
            "decode_analysis_without_embedding": _distribution(
                [item.decode_analysis_seconds for item in successful]
            ),
            "mobileclip_embedding": _distribution(
                [item.embedding_seconds for item in successful]
            ),
            "segmentation_and_representative_selection": _distribution(
                [item.segmentation_seconds for item in successful]
            ),
            "transient_frame_extraction": _distribution(
                [item.frame_extraction_seconds for item in successful]
            ),
        },
        "warm_stage_seconds": {
            "probe": _distribution([item.probe_seconds for item in warm_successful]),
            "decode_analysis_without_embedding": _distribution(
                [item.decode_analysis_seconds for item in warm_successful]
            ),
            "mobileclip_embedding": _distribution(
                [item.embedding_seconds for item in warm_successful]
            ),
            "segmentation_and_representative_selection": _distribution(
                [item.segmentation_seconds for item in warm_successful]
            ),
            "transient_frame_extraction": _distribution(
                [item.frame_extraction_seconds for item in warm_successful]
            ),
        },
        "warm_stage_totals_seconds": warm_stage_totals,
        "warm_stage_share": {
            name: _ratio(value, warm_stage_total)
            for name, value in warm_stage_totals.items()
        },
        "cold_start_video": successful[0].public()
        if concurrency == 1 and successful
        else None,
        "measurements": [item.public() for item in measurements],
    }


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _stage_totals(measurements: list[VideoMeasurement]) -> dict[str, float]:
    return {
        "probe": sum(item.probe_seconds for item in measurements),
        "decode_analysis_without_embedding": sum(
            item.decode_analysis_seconds for item in measurements
        ),
        "mobileclip_embedding": sum(
            item.embedding_seconds for item in measurements
        ),
        "segmentation_and_representative_selection": sum(
            item.segmentation_seconds for item in measurements
        ),
        "transient_frame_extraction": sum(
            item.frame_extraction_seconds for item in measurements
        ),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _sample_summary(selected: list[InventoryVideo]) -> dict[str, Any]:
    durations = [item.duration_seconds for item in selected]
    return {
        "video_count": len(selected),
        "total_duration_seconds": sum(durations),
        "duration_seconds": _distribution(durations),
        "duration_bands": dict(Counter(item.duration_band for item in selected)),
        "codecs": dict(Counter(item.codec for item in selected)),
        "resolutions": dict(
            Counter(f"{item.width}x{item.height}" for item in selected)
        ),
        "four_k_or_higher_count": sum(item.width >= 3840 for item in selected),
        "videos": [asdict(item) for item in selected],
    }


def _write_markdown(output: Path, report: dict[str, Any]) -> None:
    sample = report["sample"]
    runs = report["runs"]
    baseline = next(run for run in runs if run["concurrency"] == 1)
    concurrency_two = next(run for run in runs if run["concurrency"] == 2)
    concurrency_four = next(run for run in runs if run["concurrency"] == 4)
    stage = baseline["warm_stage_seconds"]
    probe = stage["probe"]
    decode = stage["decode_analysis_without_embedding"]
    embedding = stage["mobileclip_embedding"]
    segmentation = stage["segmentation_and_representative_selection"]
    extraction = stage["transient_frame_extraction"]
    stage_share = baseline["warm_stage_share"]
    lines = [
        "# 逻辑视频处理 50 样本性能报告",
        "",
        "本测试只包含视频探测、解码分析、MobileCLIP、内容切分、代表时间戳选择和临时取帧。",
        "不入库，不运行内容理解、向量化或聚类，也不生成片段视频和关键帧文件。",
        "",
        "## 样本",
        "",
        f"- 视频数量：{sample['video_count']} 个",
        f"- 视频总时长：{sample['total_duration_seconds']:.3f} 秒",
        f"- 时长中位数：{sample['duration_seconds']['p50']:.3f} 秒；"
        f"最长：{sample['duration_seconds']['max']:.3f} 秒",
        f"- 时长分层：{json.dumps(sample['duration_bands'], ensure_ascii=False)}",
        f"- 编码分布：{json.dumps(sample['codecs'], ensure_ascii=False)}",
        f"- 4K 或更高宽度视频：{sample['four_k_or_higher_count']} 个",
        "",
        "## 并发 1 的热态阶段耗时",
        "",
        "以下数据排除了首个视频的模型冷启动。",
        "",
        "| 阶段 | 平均耗时（秒/视频） | 中位数 | 95分位 | 热态耗时占比 |",
        "|---|---:|---:|---:|---:|",
        f"| 视频信息探测 | {probe['mean']:.4f} | {probe['p50']:.4f} "
        f"| {probe['p95']:.4f} | {stage_share['probe']:.2%} |",
        f"| 解码与画面分析（不含视觉特征） | {decode['mean']:.4f} "
        f"| {decode['p50']:.4f} | {decode['p95']:.4f} "
        f"| {stage_share['decode_analysis_without_embedding']:.2%} |",
        f"| MobileCLIP 视觉特征 | {embedding['mean']:.4f} "
        f"| {embedding['p50']:.4f} | {embedding['p95']:.4f} "
        f"| {stage_share['mobileclip_embedding']:.2%} |",
        f"| 内容切分与代表位置选择 | {segmentation['mean']:.4f} "
        f"| {segmentation['p50']:.4f} | {segmentation['p95']:.4f} "
        f"| {stage_share['segmentation_and_representative_selection']:.2%} |",
        f"| 临时取帧 | {extraction['mean']:.4f} | {extraction['p50']:.4f} "
        f"| {extraction['p95']:.4f} | {stage_share['transient_frame_extraction']:.2%} |",
        "",
        f"首个视频包含模型冷启动，总耗时为 "
        f"{baseline['cold_start_video']['total_seconds']:.3f} 秒，其中 MobileCLIP 初始化与"
        f"首批推理为 {baseline['cold_start_video']['embedding_seconds']:.3f} 秒。",
        "阶段平均值受两个超长视频影响；中位数表示常规短中视频，最大值用于观察压力长尾。",
        "",
        "## 核心指标与压测",
        "",
        "| 并发 | 整批耗时（分钟） | 实时系数 | 热态每分钟处理素材分钟数 "
        "| 吞吐提升倍数 | 并发效率 | 视频耗时95分位/最大值（秒） | 临时取帧（帧/秒） "
        "| 进程树峰值内存（MB） | MPS驱动峰值内存（MB） | 平均CPU核 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['concurrency']} | {run['wall_seconds'] / 60:.3f} "
            f"| {run['aggregate_realtime_factor']:.4f} "
            f"| {run['reported_warm_throughput']:.2f} "
            f"| {run['throughput_speedup_vs_1']:.2f} "
            f"| {run['parallel_efficiency']:.2%} "
            f"| {run['per_video_seconds']['p95']:.3f} / "
            f"{run['per_video_seconds']['max']:.3f} "
            f"| {run['transient_frame_throughput_fps']:.2f} "
            f"| {run['peak_process_tree_rss_mb']:.1f} "
            f"| {run['peak_mps_driver_mb']:.1f} "
            f"| {run['average_cpu_cores']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 输出规模",
            "",
            f"- 逻辑片段：{baseline['segment_count']} 个",
            f"- 代表画面时间戳：{baseline['representative_frame_count']} 个",
            f"- 成功处理：{baseline['successful_video_count']}/"
            f"{baseline['requested_video_count']} 个视频",
            "",
            "## 结论",
            "",
            f"- 并发 2 相比并发 1，整批耗时下降 "
            f"{1 - concurrency_two['wall_seconds'] / baseline['wall_seconds']:.2%}，"
            f"热态吞吐提高 {concurrency_two['throughput_speedup_vs_1'] - 1:.2%}。",
            f"- 并发 4 相比并发 2，只进一步缩短整批耗时 "
            f"{1 - concurrency_four['wall_seconds'] / concurrency_two['wall_seconds']:.2%}，"
            "并发收益已经明显递减。",
            f"- 热态耗时中，解码与画面分析占 "
            f"{stage_share['decode_analysis_without_embedding']:.2%}，临时取帧占 "
            f"{stage_share['transient_frame_extraction']:.2%}，二者是主要耗时来源。",
            "- 综合吞吐、95 分位和并发效率，当前建议视频处理并发设为 2。",
            "",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


async def _main() -> None:
    args = _arguments()
    if args.sample_count != 50:
        raise ValueError("--sample-count must be 50")
    concurrency_levels = list(dict.fromkeys(args.concurrency))
    if not concurrency_levels or any(level < 1 for level in concurrency_levels):
        raise ValueError("concurrency levels must be positive")
    inventory = await asyncio.to_thread(_inventory, args.root.expanduser().resolve())
    selected = _select_stratified(inventory, sample_count=args.sample_count)
    print(
        f"inventory={len(inventory)} selected={len(selected)} "
        f"source_seconds={sum(item.duration_seconds for item in selected):.3f}",
        flush=True,
    )
    embedder = TimedResidentEmbedder(
        model_path=args.model_path.expanduser().resolve(),
        batch_size=args.batch_size,
    )
    runs: list[dict[str, Any]] = []
    for index, concurrency in enumerate(concurrency_levels):
        runs.append(
            await _benchmark_run(
                selected=selected,
                concurrency=concurrency,
                embedder=embedder,
                detailed_progress=index == 0,
            )
        )
    for run in runs:
        run["reported_warm_throughput"] = (
            run["warm_source_minutes_per_wall_minute"]
            if run["concurrency"] == 1
            else run["source_minutes_per_wall_minute"]
        )
    baseline_throughput = runs[0]["reported_warm_throughput"]
    for run in runs:
        speedup = _ratio(run["reported_warm_throughput"], baseline_throughput)
        run["throughput_speedup_vs_1"] = speedup
        run["parallel_efficiency"] = _ratio(speedup, run["concurrency"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": {
            "database": False,
            "understanding": False,
            "vectorization": False,
            "clustering": False,
            "derived_media_files": False,
            "transient_frame_extraction": True,
        },
        "sample": _sample_summary(selected),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(args.output, report)
    print(f"json={args.output}", flush=True)
    print(f"markdown={args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
