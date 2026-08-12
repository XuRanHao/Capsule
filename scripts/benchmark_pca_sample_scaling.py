"""Measure how the preferred PCA dimension changes with sample count.

This benchmark is read-only: it loads indexed native vectors from existing
workspaces, repeatedly samples them without replacement, and runs the current
PCA + HDBSCAN implementation with fixed clustering parameters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from benchmark_pca_clustering import _cluster_metrics
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from capsule.config import Settings
from capsule.db.repositories import EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import EmbeddingType
from capsule.vectorstore.milvus import MilvusVectorStore

DEFAULT_DIMENSIONS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
DEFAULT_SAMPLE_COUNTS = (25, 50, 100, 200, 400, 800, 1200)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not parsed or parsed[0] < 2:
        raise argparse.ArgumentTypeError("values must be comma-separated integers >= 2")
    return parsed


def _finite(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _median(values: list[Any]) -> float | None:
    finite = [resolved for item in values if (resolved := _finite(item)) is not None]
    return float(median(finite)) if finite else None


def _mean(values: list[Any]) -> float | None:
    finite = [resolved for item in values if (resolved := _finite(item)) is not None]
    return float(fmean(finite)) if finite else None


async def _load_vectors(
    *,
    workspace_id: str,
    settings: Settings,
    repository: EmbeddingRepository,
    vector_store: MilvusVectorStore,
) -> tuple[np.ndarray, np.ndarray]:
    assets = await repository.list_indexed_cluster_embeddings(
        workspace_id=workspace_id,
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        milvus_collection=settings.milvus_collection,
    )
    assets = sorted(assets, key=lambda item: item.asset_id)
    raw_vectors = await vector_store.fetch_vectors([item.embedding_id for item in assets])
    loaded = [
        (raw_vectors[item.embedding_id], item.source_file_id)
        for item in assets
        if item.embedding_id in raw_vectors
    ]
    vectors = [vector for vector, _ in loaded]
    if not vectors:
        raise ValueError(f"workspace has no indexed native vectors: {workspace_id}")
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray([source_file_id for _, source_file_id in loaded]),
    )


def _noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    resolved = np.asarray(labels, dtype=np.int64).copy()
    next_label = int(np.max(resolved, initial=-1)) + 1
    for index in np.flatnonzero(resolved == -1):
        resolved[index] = next_label + int(index)
    return resolved


def _source_pairwise_metrics(
    source_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, float | None]:
    source_counts: dict[str, int] = defaultdict(int)
    cluster_counts: dict[int, int] = defaultdict(int)
    intersection_counts: dict[tuple[int, str], int] = defaultdict(int)
    for source, cluster in zip(source_labels.tolist(), cluster_labels.tolist(), strict=True):
        source_counts[str(source)] += 1
        if cluster == -1:
            continue
        cluster_counts[int(cluster)] += 1
        intersection_counts[(int(cluster), str(source))] += 1

    choose_two = lambda count: count * (count - 1) // 2  # noqa: E731
    true_pairs = sum(choose_two(count) for count in source_counts.values())
    predicted_pairs = sum(choose_two(count) for count in cluster_counts.values())
    true_positive = sum(choose_two(count) for count in intersection_counts.values())
    precision = true_positive / predicted_pairs if predicted_pairs else None
    recall = true_positive / true_pairs if true_pairs else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "source_pairwise_precision": precision,
        "source_pairwise_recall": recall,
        "source_pairwise_f1": f1,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["workspace_id"], row["sample_count"], row["requested_pca_dimension"])].append(
            row
        )

    fields = (
        "cluster_count",
        "coverage",
        "noise_ratio",
        "pipeline_quality_score",
        "original_cosine_silhouette",
        "final_coverage_aware_silhouette",
        "final_centroid_margin_mean",
        "final_negative_margin_ratio",
        "final_max_cluster_share",
        "source_ari_noise_singleton",
        "source_nmi_noise_singleton",
        "source_pairwise_precision",
        "source_pairwise_recall",
        "source_pairwise_f1",
        "cluster_runtime_ms",
    )
    aggregates: list[dict[str, Any]] = []
    for (workspace_id, sample_count, dimension), items in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "workspace_id": workspace_id,
            "sample_count": sample_count,
            "requested_pca_dimension": dimension,
            "effective_pca_dimension": items[0]["effective_pca_dimension"],
            "repeat_count": len(items),
        }
        for field in fields:
            values = [item[field] for item in items]
            aggregate[f"median_{field}"] = _median(values)
            aggregate[f"mean_{field}"] = _mean(values)
        aggregates.append(aggregate)
    return aggregates


def _best_dimensions(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        grouped[(row["workspace_id"], row["sample_count"])].append(row)

    selected: list[dict[str, Any]] = []
    for (workspace_id, sample_count), rows in sorted(grouped.items()):
        # Same-source pairwise F1 is weak supervision available in the document
        # benchmark workspaces. Coverage-aware silhouette breaks ties.
        best = max(
            rows,
            key=lambda row: (
                row["median_source_pairwise_f1"] or -1.0,
                row["median_final_coverage_aware_silhouette"] or -1.0,
            ),
        )
        selected.append(
            {
                "workspace_id": workspace_id,
                "sample_count": sample_count,
                "best_pca_dimension": best["requested_pca_dimension"],
                "effective_pca_dimension": best["effective_pca_dimension"],
                "median_source_pairwise_f1": best["median_source_pairwise_f1"],
                "median_source_ari_noise_singleton": best[
                    "median_source_ari_noise_singleton"
                ],
                "median_coverage_aware_silhouette": best[
                    "median_final_coverage_aware_silhouette"
                ],
                "median_pipeline_quality_score": best["median_pipeline_quality_score"],
                "median_coverage": best["median_coverage"],
                "median_cluster_count": best["median_cluster_count"],
            }
        )
    return selected


async def _run(
    *,
    workspace_ids: tuple[str, ...],
    dimensions: tuple[int, ...],
    sample_counts: tuple[int, ...],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    settings = Settings()
    database = Database(settings)
    try:
        repository = EmbeddingRepository(database)
        vector_store = MilvusVectorStore(settings)
        datasets = {
            workspace_id: await _load_vectors(
                workspace_id=workspace_id,
                settings=settings,
                repository=repository,
                vector_store=vector_store,
            )
            for workspace_id in workspace_ids
        }
        rows: list[dict[str, Any]] = []
        for dataset_index, (workspace_id, dataset) in enumerate(datasets.items()):
            vectors, source_labels = dataset
            available_counts = tuple(count for count in sample_counts if count <= len(vectors))
            if not available_counts:
                raise ValueError(
                    f"workspace {workspace_id} has {len(vectors)} vectors, below all sample counts"
                )
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + dataset_index * 10_000 + repeat)
                permutation = rng.permutation(len(vectors))
                for sample_count in available_counts:
                    indices = np.sort(permutation[:sample_count])
                    sample = vectors[indices]
                    sample_sources = source_labels[indices]
                    for dimension in dimensions:
                        result = _cluster_metrics(sample, dimension, extended=False)
                        cluster_labels = np.asarray(result["final_labels"], dtype=np.int64)
                        singleton_labels = _noise_as_singletons(cluster_labels)
                        result.update(
                            {
                                "source_ari_noise_singleton": float(
                                    adjusted_rand_score(sample_sources, singleton_labels)
                                ),
                                "source_nmi_noise_singleton": float(
                                    normalized_mutual_info_score(sample_sources, singleton_labels)
                                ),
                                **_source_pairwise_metrics(sample_sources, cluster_labels),
                            }
                        )
                        result.pop("raw_labels")
                        result.pop("final_labels")
                        result.update(
                            {
                                "workspace_id": workspace_id,
                                "repeat": repeat,
                            }
                        )
                        rows.append(result)
        aggregates = _aggregate(rows)
        return {
            "metadata": {
                "workspace_vector_counts": {
                    workspace_id: len(dataset[0])
                    for workspace_id, dataset in datasets.items()
                },
                "workspace_source_counts": {
                    workspace_id: len(set(dataset[1].tolist()))
                    for workspace_id, dataset in datasets.items()
                },
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "pca_dimensions": list(dimensions),
                "sample_counts": list(sample_counts),
                "repeats": repeats,
                "seed": seed,
                "hdbscan": {
                    "min_cluster_size": 3,
                    "min_samples": 3,
                    "cluster_selection_method": "eom",
                    "cluster_selection_epsilon": 0.5,
                },
                "selection_rule": (
                    "maximize median same-source pairwise F1; break ties with median "
                    "coverage-aware original-space cosine silhouette"
                ),
            },
            "best_dimensions": _best_dimensions(aggregates),
            "aggregates": aggregates,
            "runs": rows,
        }
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-id", action="append", required=True)
    parser.add_argument("--dimensions", type=_parse_ints, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--sample-counts", type=_parse_ints, default=DEFAULT_SAMPLE_COUNTS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    report = asyncio.run(
        _run(
            workspace_ids=tuple(args.workspace_id),
            dimensions=args.dimensions,
            sample_counts=args.sample_counts,
            repeats=args.repeats,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
