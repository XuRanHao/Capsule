"""Benchmark PCA target dimensions against the current clustering pipeline.

The benchmark reads the latest indexed raw vectors, builds the same 0.3 native
+ 0.7 feature fusion used by production clustering, and never mutates Postgres
or Milvus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np
from hdbscan.validity import validity_index
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from capsule.config import Settings
from capsule.db.repositories import EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import EmbeddingType
from capsule.pipeline.clustering import (
    HdbscanParameters,
    cluster_vectors,
    merge_semantically_overlapping_clusters,
)
from capsule.vector_fusion import (
    DEFAULT_NATIVE_CONTENT_WEIGHT,
    fuse_native_dimension_vectors,
)
from capsule.vectorstore.milvus import MilvusVectorStore

PCA_DIMENSIONS = (2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64)
SAMPLE_COUNTS = (15, 25, 35, 40, 45, 50)
RESAMPLE_REPEATS = 12
FULL_SAMPLE_CHANNELS = (
    EmbeddingType.NATIVE_MULTIMODAL,
    EmbeddingType.ASSET_DESCRIPTION,
    EmbeddingType.ASSET_USAGE,
    EmbeddingType.COLOR_COMPOSITION,
    EmbeddingType.MOOD_ATMOSPHERE,
    EmbeddingType.SUBJECT_CONTENT,
    EmbeddingType.VISUAL_STYLE,
)
BOUNDARY_CHANNELS = (
    EmbeddingType.SCENE_THEME,
    EmbeddingType.CHARACTER_STATE_OR_PSYCHOLOGY,
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / norms, dtype=np.float32)


def _finite(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(median(finite)) if finite else None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(fmean(finite)) if finite else None


def _noise_as_singletons(labels: np.ndarray) -> np.ndarray:
    """Keep every noise point independent instead of treating -1 as one cluster."""
    resolved = np.asarray(labels, dtype=np.int64).copy()
    next_label = int(np.max(resolved, initial=-1)) + 1
    for index in np.flatnonzero(resolved == -1):
        resolved[index] = next_label + int(index)
    return resolved


def _pairwise_coassignment_f1(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    row, column = np.triu_indices(len(reference), k=1)
    reference_pairs = (reference[row] == reference[column]) & (reference[row] != -1)
    candidate_pairs = (candidate[row] == candidate[column]) & (candidate[row] != -1)
    true_positive = int(np.count_nonzero(reference_pairs & candidate_pairs))
    false_positive = int(np.count_nonzero(~reference_pairs & candidate_pairs))
    false_negative = int(np.count_nonzero(reference_pairs & ~candidate_pairs))
    denominator = 2 * true_positive + false_positive + false_negative
    return None if denominator == 0 else 2 * true_positive / denominator


def _noise_jaccard(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_noise = reference == -1
    candidate_noise = candidate == -1
    union = int(np.count_nonzero(reference_noise | candidate_noise))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(reference_noise & candidate_noise)) / union


def _final_structure_metrics(
    normalized: np.ndarray,
    labels: np.ndarray,
    *,
    compute_dbcv: bool,
) -> dict[str, float | None]:
    clustered = labels != -1
    cluster_labels = sorted(set(labels.tolist()) - {-1})
    cluster_count = len(cluster_labels)
    clustered_count = int(np.count_nonzero(clustered))
    coverage = clustered_count / len(labels)
    silhouette: float | None = None
    dbcv: float | None = None
    margin_mean: float | None = None
    margin_p10: float | None = None
    negative_margin_ratio: float | None = None
    if cluster_count >= 2 and clustered_count > cluster_count:
        silhouette = _finite(
            silhouette_score(normalized[clustered], labels[clustered], metric="cosine")
        )
        if compute_dbcv:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dbcv = _finite(
                        validity_index(
                            np.asarray(normalized, dtype=np.float64),
                            labels,
                            metric="euclidean",
                        )
                    )
            except (ArithmeticError, ValueError):
                dbcv = None
        centroids = {}
        for label in cluster_labels:
            centroid = np.mean(normalized[labels == label], axis=0)
            centroids[label] = centroid / np.linalg.norm(centroid)
        margins: list[float] = []
        for index in np.flatnonzero(clustered):
            label = int(labels[index])
            own = float(normalized[index] @ centroids[label])
            other = max(
                float(normalized[index] @ centroids[candidate])
                for candidate in cluster_labels
                if candidate != label
            )
            margins.append(own - other)
        margin_mean = float(np.mean(margins))
        margin_p10 = float(np.percentile(margins, 10))
        negative_margin_ratio = float(np.mean(np.asarray(margins) < 0.0))

    sizes = np.asarray([np.count_nonzero(labels == label) for label in cluster_labels])
    max_cluster_share: float | None = None
    small_cluster_asset_share: float | None = None
    normalized_entropy: float | None = None
    if clustered_count:
        proportions = sizes / clustered_count
        max_cluster_share = float(np.max(proportions))
        small_cluster_asset_share = float(np.sum(sizes[sizes <= 6]) / clustered_count)
        if cluster_count >= 2:
            normalized_entropy = float(
                -np.sum(proportions * np.log(proportions)) / np.log(cluster_count)
            )
    coverage_aware_silhouette = (
        coverage * (silhouette + 1.0) / 2.0 if silhouette is not None else 0.0
    )
    return {
        "final_original_cosine_silhouette": silhouette,
        "final_coverage_aware_silhouette": coverage_aware_silhouette,
        "final_original_dbcv": dbcv,
        "final_centroid_margin_mean": margin_mean,
        "final_centroid_margin_p10": margin_p10,
        "final_negative_margin_ratio": negative_margin_ratio,
        "final_max_cluster_share": max_cluster_share,
        "final_small_cluster_asset_share": small_cluster_asset_share,
        "final_normalized_cluster_entropy": normalized_entropy,
    }


def _cluster_metrics(
    vectors: np.ndarray,
    requested_dimension: int,
    *,
    extended: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = cluster_vectors(
        vectors,
        pca_dimension=requested_dimension,
        parameters=HdbscanParameters(
            min_cluster_size=3,
            min_samples=3,
            cluster_selection_epsilon=0.5,
        ),
    )
    cluster_ms = (time.perf_counter() - started) * 1000
    labels = result.labels
    clustered = labels != -1
    cluster_count = result.cluster_count
    normalized = _normalize(vectors)

    target_dimension = result.pca_dimension
    pca_fit_transform_ms: float | None = None
    explained_variance: float | None = None
    neighborhood_trustworthiness: float | None = None
    distance_correlation: float | None = None
    if extended:
        pca_started = time.perf_counter()
        pca = PCA(n_components=target_dimension, random_state=0)
        projection = pca.fit_transform(normalized)
        pca_fit_transform_ms = (time.perf_counter() - pca_started) * 1000
        explained_variance = float(np.sum(pca.explained_variance_ratio_))
        neighbors = min(5, max(1, (len(vectors) - 1) // 2))
        neighborhood_trustworthiness = float(
            trustworthiness(normalized, projection, n_neighbors=neighbors, metric="cosine")
        )
        original_distances = pdist(normalized, metric="cosine")
        projected_distances = pdist(result.transformed_vectors, metric="cosine")
        distance_correlation = _finite(
            spearmanr(original_distances, projected_distances).statistic
        )

    projected_silhouette: float | None = None
    original_silhouette: float | None = None
    calinski_harabasz: float | None = None
    davies_bouldin: float | None = None
    dbcv: float | None = None
    if cluster_count >= 2 and int(np.count_nonzero(clustered)) > cluster_count:
        projected_silhouette = _finite(
            silhouette_score(result.transformed_vectors[clustered], labels[clustered])
        )
        original_silhouette = _finite(
            silhouette_score(normalized[clustered], labels[clustered], metric="cosine")
        )
        calinski_harabasz = _finite(
            calinski_harabasz_score(normalized[clustered], labels[clustered])
        )
        davies_bouldin = _finite(
            davies_bouldin_score(normalized[clustered], labels[clustered])
        )
        if extended:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    dbcv = _finite(
                        validity_index(
                            np.asarray(result.transformed_vectors, dtype=np.float64),
                            labels,
                            metric="euclidean",
                        )
                    )
            except (ArithmeticError, ValueError):
                dbcv = None

    merged = merge_semantically_overlapping_clusters(vectors, labels)
    final_metrics = _final_structure_metrics(
        normalized,
        merged.labels,
        compute_dbcv=extended,
    )
    clustered_probabilities = result.probabilities[clustered]
    return {
        "requested_pca_dimension": requested_dimension,
        "effective_pca_dimension": target_dimension,
        "sample_count": len(vectors),
        "cluster_count": cluster_count,
        "merged_cluster_count": merged.cluster_count,
        "semantic_merge_count": len(merged.decisions),
        "noise_ratio": result.noise_ratio,
        "coverage": 1.0 - result.noise_ratio,
        "mean_membership_probability": (
            float(np.mean(clustered_probabilities)) if len(clustered_probabilities) else None
        ),
        "pipeline_quality_score": _finite(result.quality_score),
        "projected_silhouette": projected_silhouette,
        "original_cosine_silhouette": original_silhouette,
        "dbcv": dbcv,
        "calinski_harabasz_original": calinski_harabasz,
        "davies_bouldin_original": davies_bouldin,
        "explained_variance": explained_variance,
        "trustworthiness": neighborhood_trustworthiness,
        "pairwise_distance_spearman": distance_correlation,
        "pca_fit_transform_ms": pca_fit_transform_ms,
        "cluster_runtime_ms": cluster_ms,
        "raw_labels": labels.tolist(),
        "final_labels": merged.labels.tolist(),
        **final_metrics,
    }


async def _load_channel_vectors(
    *,
    settings: Settings,
    repository: EmbeddingRepository,
    vector_store: MilvusVectorStore,
    workspace_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    native_assets = await repository.list_indexed_cluster_embeddings(
        workspace_id=workspace_id,
        embedding_type=EmbeddingType.NATIVE_MULTIMODAL.value,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        milvus_collection=settings.milvus_collection,
    )
    native_by_asset = {asset.asset_id: asset for asset in native_assets}
    channels: dict[str, np.ndarray] = {}
    asset_ids: dict[str, list[str]] = {}
    selected_channels = (*FULL_SAMPLE_CHANNELS, *BOUNDARY_CHANNELS)
    for embedding_type in selected_channels:
        assets = await repository.list_indexed_cluster_embeddings(
            workspace_id=workspace_id,
            embedding_type=embedding_type.value,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            milvus_collection=settings.milvus_collection,
        )
        assets = sorted(assets, key=lambda item: item.asset_id)
        vector_ids = {asset.embedding_id for asset in assets}
        if embedding_type is not EmbeddingType.NATIVE_MULTIMODAL:
            vector_ids.update(
                native_by_asset[asset.asset_id].embedding_id
                for asset in assets
                if asset.asset_id in native_by_asset
            )
        raw_vectors = await vector_store.fetch_vectors(sorted(vector_ids))
        loaded_vectors: list[list[float]] = []
        loaded_asset_ids: list[str] = []
        for asset in assets:
            dimension_vector = raw_vectors.get(asset.embedding_id)
            if dimension_vector is None:
                continue
            if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
                vector = dimension_vector
            else:
                native_asset = native_by_asset.get(asset.asset_id)
                native_vector = (
                    raw_vectors.get(native_asset.embedding_id)
                    if native_asset is not None
                    else None
                )
                if native_vector is None:
                    continue
                vector = fuse_native_dimension_vectors(
                    native_vector=native_vector,
                    dimension_vector=dimension_vector,
                    native_content_weight=DEFAULT_NATIVE_CONTENT_WEIGHT,
                )
            loaded_asset_ids.append(asset.asset_id)
            loaded_vectors.append(vector)
        if loaded_vectors:
            channels[embedding_type.value] = np.asarray(loaded_vectors, dtype=np.float32)
            asset_ids[embedding_type.value] = loaded_asset_ids
    return channels, asset_ids


def _aggregate_full_results(full_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in full_results:
        if item["sample_count"] == 50:
            grouped[item["requested_pca_dimension"]].append(item)
    aggregates: list[dict[str, Any]] = []
    fields = (
        "cluster_count",
        "merged_cluster_count",
        "noise_ratio",
        "coverage",
        "mean_membership_probability",
        "pipeline_quality_score",
        "projected_silhouette",
        "original_cosine_silhouette",
        "dbcv",
        "calinski_harabasz_original",
        "davies_bouldin_original",
        "explained_variance",
        "trustworthiness",
        "pairwise_distance_spearman",
        "cluster_runtime_ms",
        "pca_fit_transform_ms",
        "semantic_merge_count",
        "final_original_cosine_silhouette",
        "final_coverage_aware_silhouette",
        "final_original_dbcv",
        "final_centroid_margin_mean",
        "final_centroid_margin_p10",
        "final_negative_margin_ratio",
        "final_max_cluster_share",
        "final_small_cluster_asset_share",
        "final_normalized_cluster_entropy",
    )
    for dimension in PCA_DIMENSIONS:
        items = grouped.get(dimension, [])
        if not items:
            continue
        aggregate: dict[str, Any] = {
            "requested_pca_dimension": dimension,
            "channel_count": len(items),
        }
        for field in fields:
            aggregate[f"median_{field}"] = _median([item[field] for item in items])
            aggregate[f"mean_{field}"] = _mean([item[field] for item in items])
        aggregates.append(aggregate)
    return aggregates


def _resample(
    channels: dict[str, np.ndarray],
    full_by_channel_dimension: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260811)
    measurements: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    full_channels = {
        name: vectors for name, vectors in channels.items() if len(vectors) == 50
    }
    for channel, vectors in full_channels.items():
        for dimension in PCA_DIMENSIONS:
            reference = np.asarray(
                full_by_channel_dimension[(channel, dimension)]["final_labels"], dtype=np.int_
            )
            for sample_count in SAMPLE_COUNTS:
                repeats = 1 if sample_count == len(vectors) else RESAMPLE_REPEATS
                for _ in range(repeats):
                    indices = (
                        np.arange(len(vectors))
                        if sample_count == len(vectors)
                        else np.sort(rng.choice(len(vectors), size=sample_count, replace=False))
                    )
                    result = _cluster_metrics(vectors[indices], dimension, extended=False)
                    result.pop("raw_labels")
                    labels = np.asarray(result.pop("final_labels"), dtype=np.int_)
                    reference_subset = reference[indices]
                    singleton_reference = _noise_as_singletons(reference_subset)
                    singleton_labels = _noise_as_singletons(labels)
                    shared_clustered = (labels != -1) & (reference_subset != -1)
                    ari_clustered = None
                    nmi_clustered = None
                    if int(np.count_nonzero(shared_clustered)) >= 3:
                        ari_clustered = _finite(
                            adjusted_rand_score(
                                reference_subset[shared_clustered], labels[shared_clustered]
                            )
                        )
                        nmi_clustered = _finite(
                            normalized_mutual_info_score(
                                reference_subset[shared_clustered], labels[shared_clustered]
                            )
                        )
                    measurements[(dimension, sample_count)].append(
                        {
                            "ari_all": _finite(adjusted_rand_score(reference_subset, labels)),
                            "nmi_all": _finite(
                                normalized_mutual_info_score(reference_subset, labels)
                            ),
                            "ari_noise_singleton": _finite(
                                adjusted_rand_score(singleton_reference, singleton_labels)
                            ),
                            "nmi_noise_singleton": _finite(
                                normalized_mutual_info_score(
                                    singleton_reference, singleton_labels
                                )
                            ),
                            "pairwise_coassignment_f1": _pairwise_coassignment_f1(
                                reference_subset, labels
                            ),
                            "noise_jaccard": _noise_jaccard(reference_subset, labels),
                            "ari_shared_clustered": ari_clustered,
                            "nmi_shared_clustered": nmi_clustered,
                            "noise_ratio": result["noise_ratio"],
                            "coverage": result["coverage"],
                            "pipeline_quality_score": result["pipeline_quality_score"],
                            "cluster_count": result["cluster_count"],
                            "effective_pca_dimension": result["effective_pca_dimension"],
                            "cluster_runtime_ms": result["cluster_runtime_ms"],
                            "structured": float(result["cluster_count"] >= 2),
                        }
                    )

    aggregates: list[dict[str, Any]] = []
    fields = (
        "ari_all",
        "nmi_all",
        "ari_shared_clustered",
        "nmi_shared_clustered",
        "ari_noise_singleton",
        "nmi_noise_singleton",
        "pairwise_coassignment_f1",
        "noise_jaccard",
        "noise_ratio",
        "coverage",
        "pipeline_quality_score",
        "cluster_count",
        "effective_pca_dimension",
        "cluster_runtime_ms",
        "structured",
    )
    for (dimension, sample_count), items in sorted(measurements.items()):
        aggregate: dict[str, Any] = {
            "requested_pca_dimension": dimension,
            "sample_count": sample_count,
            "run_count": len(items),
        }
        for field in fields:
            values = [item[field] for item in items]
            aggregate[f"median_{field}"] = _median(values)
            aggregate[f"mean_{field}"] = _mean(values)
        aggregates.append(aggregate)
    return aggregates


async def _run(workspace_id: str) -> dict[str, Any]:
    settings = Settings()
    database = Database(settings)
    try:
        repository = EmbeddingRepository(database)
        vector_store = MilvusVectorStore(settings)
        channels, asset_ids = await _load_channel_vectors(
            settings=settings,
            repository=repository,
            vector_store=vector_store,
            workspace_id=workspace_id,
        )
        full_results: list[dict[str, Any]] = []
        full_index: dict[tuple[str, int], dict[str, Any]] = {}
        for channel, vectors in channels.items():
            for dimension in PCA_DIMENSIONS:
                result = _cluster_metrics(vectors, dimension)
                result["embedding_type"] = channel
                full_results.append(result)
                full_index[(channel, dimension)] = result
        resampling = _resample(channels, full_index)
        return {
            "metadata": {
                "workspace_id": workspace_id,
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "native_content_weight": DEFAULT_NATIVE_CONTENT_WEIGHT,
                "dimension_content_weight": 1.0 - DEFAULT_NATIVE_CONTENT_WEIGHT,
                "pca_dimensions": list(PCA_DIMENSIONS),
                "sample_counts": list(SAMPLE_COUNTS),
                "resample_repeats": RESAMPLE_REPEATS,
                "hdbscan": {
                    "min_cluster_size": 3,
                    "min_samples": 3,
                    "cluster_selection_method": "eom",
                    "cluster_selection_epsilon": 0.5,
                },
                "channel_sample_counts": {
                    channel: len(vectors) for channel, vectors in channels.items()
                },
                "channel_asset_counts": {
                    channel: len(ids) for channel, ids in asset_ids.items()
                },
            },
            "full_results": full_results,
            "full_50_aggregate": _aggregate_full_results(full_results),
            "resampling_aggregate": resampling,
        }
    finally:
        await database.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(_run(args.workspace_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
