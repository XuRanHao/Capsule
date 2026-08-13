"""Time-constrained clustering shared by video and audio parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.sparse import diags  # type: ignore[import-untyped]
from sklearn.cluster import AgglomerativeClustering


@dataclass(frozen=True, slots=True)
class TemporalRegion:
    """One contiguous run of transient feature windows."""

    atom_indices: tuple[int, ...]
    sample_indices: tuple[int, ...]
    start_ms: int
    end_ms: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000

    def centroid(self, embeddings: np.ndarray) -> np.ndarray:
        value = embeddings[list(self.sample_indices)].mean(axis=0)
        return cast(
            np.ndarray,
            value / max(float(np.linalg.norm(value)), np.finfo(np.float32).eps),
        )


def cluster_contiguous_windows(
    embeddings: np.ndarray,
    *,
    sample_timestamps_ms: list[int],
    duration_ms: int,
    minimum_segment_seconds: float,
    distance_quantile: float,
    minimum_distance_threshold: float,
    maximum_distance_threshold: float,
) -> tuple[list[TemporalRegion], float]:
    """Cluster only adjacent windows, then merge regions below the minimum.

    ``sample_timestamps_ms`` contains the center of each independent feature
    window. The returned regions cover ``[0, duration_ms]`` exactly and no
    maximum duration or desired segment count is imposed.
    """

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("temporal clustering requires a non-empty embedding matrix")
    if len(values) != len(sample_timestamps_ms):
        raise ValueError("temporal timestamps and embeddings must have equal length")
    if duration_ms <= 0 or minimum_segment_seconds <= 0:
        raise ValueError("temporal duration and minimum segment duration must be positive")
    if any(
        right <= left
        for left, right in zip(sample_timestamps_ms[:-1], sample_timestamps_ms[1:], strict=True)
    ):
        raise ValueError("temporal sample timestamps must be strictly increasing")
    if len(values) == 1:
        return [TemporalRegion((0,), (0,), 0, duration_ms)], minimum_distance_threshold

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, np.finfo(np.float32).eps)
    zero_rows = np.flatnonzero(norms[:, 0] <= np.finfo(np.float32).eps)
    if len(zero_rows):
        # Cosine linkage rejects zero vectors. A shared unit sentinel makes
        # independent silent windows mutually identical without inventing
        # boundaries or persisting a special feature. Its private extra axis is
        # orthogonal to every real embedding dimension.
        normalized = np.pad(normalized, ((0, 0), (0, 1)))
        normalized[zero_rows, -1] = 1
    distances = 1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1)
    threshold = float(
        np.clip(
            np.quantile(distances, distance_quantile),
            minimum_distance_threshold,
            maximum_distance_threshold,
        )
    )
    size = len(normalized)
    connectivity = diags(
        [np.ones(size - 1), np.ones(size - 1)],
        [-1, 1],
        shape=(size, size),
    )
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
        connectivity=connectivity,
        compute_full_tree=True,
    ).fit_predict(normalized)
    starts, ends = _runs(labels)
    boundaries = [0]
    for start in starts[1:]:
        left = sample_timestamps_ms[int(start) - 1]
        right = sample_timestamps_ms[int(start)]
        boundaries.append(round((left + right) / 2))
    boundaries.append(duration_ms)
    regions = [
        TemporalRegion(
            atom_indices=(index,),
            sample_indices=tuple(range(int(start), int(end))),
            start_ms=boundaries[index],
            end_ms=boundaries[index + 1],
        )
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]
    regions = merge_short_regions(regions, normalized, minimum_segment_seconds)
    return [
        TemporalRegion((index,), region.sample_indices, region.start_ms, region.end_ms)
        for index, region in enumerate(regions)
    ], threshold


def merge_short_regions(
    regions: list[TemporalRegion],
    embeddings: np.ndarray,
    minimum_seconds: float,
) -> list[TemporalRegion]:
    """Merge each short region into its most similar immediate neighbor."""

    result = regions.copy()
    while len(result) > 1:
        short = next(
            (
                index
                for index, region in enumerate(result)
                if region.duration_seconds < minimum_seconds
            ),
            None,
        )
        if short is None:
            return result
        center = result[short].centroid(embeddings)
        choices: list[tuple[float, int]] = []
        if short:
            choices.append(
                (1.0 - float(center @ result[short - 1].centroid(embeddings)), short - 1)
            )
        if short + 1 < len(result):
            choices.append(
                (1.0 - float(center @ result[short + 1].centroid(embeddings)), short + 1)
            )
        neighbor = min(choices)[1]
        left_index, right_index = sorted((short, neighbor))
        left, right = result[left_index], result[right_index]
        result[left_index : right_index + 1] = [
            TemporalRegion(
                atom_indices=left.atom_indices + right.atom_indices,
                sample_indices=left.sample_indices + right.sample_indices,
                start_ms=left.start_ms,
                end_ms=right.end_ms,
            )
        ]
    return result


def _runs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    return starts, np.r_[starts[1:], len(labels)]
