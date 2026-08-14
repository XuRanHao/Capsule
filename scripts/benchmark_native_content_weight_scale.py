"""Measure native-content fusion-weight sensitivity across nested COCO scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mobileclip
import numpy as np
from benchmark_rrf_k_scale import (
    CHANNEL_WEIGHTS,
    _bm25_rankings,
    _encode_texts,
    _load_annotations,
    _metrics,
    _nested_order,
    _top_rankings,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/benchmarks/coco-val2017/annotations_trainval2017.zip"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/models/mobileclip-s0/mobileclip_s0.pt"),
    )
    parser.add_argument(
        "--rrf-cache",
        type=Path,
        default=Path(
            "data/benchmarks/coco-val2017/mobileclip-s0-rrf-scale-1000-cache.npz"
        ),
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=Path(
            "data/benchmarks/coco-val2017/mobileclip-s0-native-weight-1000-cache.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/benchmarks/native-content-weight-scale-coco-1000-2026-08-13.json"
        ),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 250, 350, 500, 750, 1000],
    )
    parser.add_argument(
        "--native-weights",
        type=float,
        nargs="+",
        default=[item / 10 for item in range(11)],
    )
    parser.add_argument("--query-count", type=int, default=40)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--channel-limit", type=int, default=100)
    parser.add_argument("--candidate-cap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def _fuse(native: np.ndarray, dimension: np.ndarray, weight: float) -> np.ndarray:
    return _normalize(weight * _normalize(native) + (1.0 - weight) * _normalize(dimension))


def _build_or_load_raw_cache(
    args: argparse.Namespace,
    *,
    annotations: dict[str, Any],
    image_ids: list[int],
) -> dict[str, np.ndarray]:
    required = {"image_ids", "native_multimodal", "subject_raw", "scene_raw"}
    if args.raw_cache.exists():
        loaded = np.load(args.raw_cache)
        if set(loaded.files) == required and np.array_equal(
            loaded["image_ids"], np.asarray(image_ids, dtype=np.int64)
        ):
            print(f"reusing raw embedding cache {args.raw_cache}", flush=True)
            return {key: loaded[key] for key in loaded.files}

    source = np.load(args.rrf_cache)
    if not np.array_equal(source["image_ids"], np.asarray(image_ids, dtype=np.int64)):
        raise ValueError("RRF cache image order does not match this benchmark")
    native = source["native_multimodal"].astype(np.float32)
    captions: dict[int, list[str]] = annotations["captions"]
    categories: dict[int, set[str]] = annotations["categories"]
    subject_texts = [
        "objects: " + ", ".join(sorted(categories.get(image_id, set())))
        if categories.get(image_id)
        else "objects described as " + captions[image_id][1]
        for image_id in image_ids
    ]
    scene_texts = [". ".join(captions[image_id][1:]) for image_id in image_ids]
    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=str(args.checkpoint)
    )
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    cache = {
        "image_ids": np.asarray(image_ids, dtype=np.int64),
        "native_multimodal": native,
        "subject_raw": _encode_texts(
            model, tokenizer, subject_texts, batch_size=args.batch_size
        ),
        "scene_raw": _encode_texts(model, tokenizer, scene_texts, batch_size=args.batch_size),
    }
    args.raw_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_cache, **cache)
    print(f"wrote raw embedding cache {args.raw_cache}", flush=True)
    return cache


def _fuse_rankings(
    rankings: dict[str, list[int]],
    *,
    rrf_k: int,
    candidate_cap: int,
    include_text: bool,
) -> list[int]:
    scores: dict[int, float] = {}
    for channel, weight in CHANNEL_WEIGHTS.items():
        for rank, index in enumerate(rankings[channel], start=1):
            scores[index] = scores.get(index, 0.0) + weight / (rrf_k + rank)
    if include_text:
        for rank, index in enumerate(rankings["local_text"], start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda index: (-scores[index], index))[:candidate_cap]


def _evaluate_size(
    args: argparse.Namespace,
    *,
    size: int,
    cache: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    query_ids: list[int],
    query_texts: list[str],
    document_texts: list[str],
) -> dict[str, Any]:
    corpus_ids = cache["image_ids"][:size].astype(np.int64).tolist()
    corpus_position = {image_id: index for index, image_id in enumerate(corpus_ids)}
    text_rankings = _bm25_rankings(
        query_texts=query_texts,
        document_texts=document_texts[:size],
        limit=args.channel_limit,
    )
    results: dict[str, Any] = {}
    for native_weight in args.native_weights:
        corpus_vectors = {
            "native_multimodal": cache["native_multimodal"][:size],
            "subject_content": _fuse(
                cache["native_multimodal"][:size],
                cache["subject_raw"][:size],
                native_weight,
            ),
            "scene_theme": _fuse(
                cache["native_multimodal"][:size],
                cache["scene_raw"][:size],
                native_weight,
            ),
        }
        effective_queries = {
            "native_multimodal": query_vectors["native_multimodal"],
            "subject_content": _fuse(
                query_vectors["native_multimodal"],
                query_vectors["subject_raw"],
                native_weight,
            ),
            "scene_theme": _fuse(
                query_vectors["native_multimodal"],
                query_vectors["scene_raw"],
                native_weight,
            ),
        }
        rankings = {
            channel: _top_rankings(
                effective_queries[channel] @ corpus_vectors[channel].T,
                limit=args.channel_limit,
            )
            for channel in CHANNEL_WEIGHTS
        }
        rankings["local_text"] = text_rankings
        weight_report: dict[str, Any] = {}
        for mode, include_text in (("vector_only", False), ("vector_plus_text", True)):
            relevant_ranks: list[int | None] = []
            for query_index, query_id in enumerate(query_ids):
                fused = _fuse_rankings(
                    {channel: items[query_index] for channel, items in rankings.items()},
                    rrf_k=args.rrf_k,
                    candidate_cap=args.candidate_cap,
                    include_text=include_text,
                )
                relevant = corpus_position[query_id]
                try:
                    relevant_ranks.append(fused.index(relevant) + 1)
                except ValueError:
                    relevant_ranks.append(None)
            weight_report[mode] = _metrics(
                relevant_ranks, candidate_cap=args.candidate_cap
            )
            weight_report[f"{mode}_per_query_rank"] = relevant_ranks
        results[f"{native_weight:.1f}"] = weight_report
    return results


def main() -> None:
    args = _parse_args()
    annotations = _load_annotations(args.annotations)
    maximum_size = max(args.sizes)
    image_ids = _nested_order(
        list(annotations["file_names"]), seed=args.seed
    )[:maximum_size]
    if args.query_count > min(args.sizes):
        raise ValueError("query count must not exceed the smallest corpus")
    cache = _build_or_load_raw_cache(args, annotations=annotations, image_ids=image_ids)
    captions: dict[int, list[str]] = annotations["captions"]
    categories: dict[int, set[str]] = annotations["categories"]
    query_ids = image_ids[: args.query_count]
    query_texts = [captions[image_id][0] for image_id in query_ids]
    document_texts = [
        ". ".join(
            [
                *captions[image_id][1:],
                "objects: " + ", ".join(sorted(categories.get(image_id, set()))),
            ]
        )
        for image_id in image_ids
    ]

    model, _, _ = mobileclip.create_model_and_transforms(
        "mobileclip_s0", pretrained=str(args.checkpoint)
    )
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    query_vectors = {
        "native_multimodal": _encode_texts(
            model, tokenizer, query_texts, batch_size=args.batch_size
        ),
        "subject_raw": _encode_texts(
            model,
            tokenizer,
            [f"objects and people: {text}" for text in query_texts],
            batch_size=args.batch_size,
        ),
        "scene_raw": _encode_texts(
            model,
            tokenizer,
            [f"scene and setting: {text}" for text in query_texts],
            batch_size=args.batch_size,
        ),
    }

    scales: dict[str, Any] = {}
    for size in args.sizes:
        scales[str(size)] = _evaluate_size(
            args,
            size=size,
            cache=cache,
            query_vectors=query_vectors,
            query_ids=query_ids,
            query_texts=query_texts,
            document_texts=document_texts,
        )
        best = max(
            args.native_weights,
            key=lambda weight: scales[str(size)][f"{weight:.1f}"]["vector_plus_text"][
                "ndcg_at_10"
            ],
        )
        print(f"size={size}: best native weight={best:.1f}", flush=True)

    report = {
        "dataset": "COCO val2017",
        "embedding_model": "MobileCLIP-S0 (local checkpoint)",
        "seed": args.seed,
        "query_count": args.query_count,
        "query_image_ids": query_ids,
        "sizes": args.sizes,
        "native_content_weights": args.native_weights,
        "rrf_k": args.rrf_k,
        "channel_limit": args.channel_limit,
        "candidate_cap": args.candidate_cap,
        "dimension_weights": CHANNEL_WEIGHTS,
        "fusion_scope": "both corpus and query vectors",
        "query_dimension_method": "fixed prompt prefixes; no ground-truth category leakage",
        "scales": scales,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote report {args.output}", flush=True)


if __name__ == "__main__":
    main()
