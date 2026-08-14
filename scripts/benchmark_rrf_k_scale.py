"""Measure weighted-RRF K sensitivity while growing a nested COCO corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import mobileclip
import numpy as np
import torch
from PIL import Image

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
    }
)
CHANNEL_WEIGHTS = {
    "native_multimodal": 0.4,
    "subject_content": 0.3,
    "scene_theme": 0.3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/benchmarks/coco-val2017/val2017.zip"),
    )
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
        "--cache",
        type=Path,
        default=Path(
            "data/benchmarks/coco-val2017/mobileclip-s0-rrf-scale-1000-cache.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/rrf-k-scale-coco-1000-2026-08-13.json"),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 1000],
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 30, 60, 90, 120],
    )
    parser.add_argument("--query-count", type=int, default=40)
    parser.add_argument("--channel-limit", type=int, default=100)
    parser.add_argument("--candidate-cap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def _load_annotations(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        captions = json.loads(
            archive.read("annotations/captions_val2017.json").decode("utf-8")
        )
        instances = json.loads(
            archive.read("annotations/instances_val2017.json").decode("utf-8")
        )

    file_names = {int(item["id"]): item["file_name"] for item in captions["images"]}
    captions_by_image: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for item in captions["annotations"]:
        captions_by_image[int(item["image_id"])].append(
            (int(item["id"]), str(item["caption"]).strip())
        )
    ordered_captions = {
        image_id: [caption for _, caption in sorted(items)]
        for image_id, items in captions_by_image.items()
    }

    category_names = {
        int(item["id"]): str(item["name"]) for item in instances["categories"]
    }
    categories_by_image: dict[int, set[str]] = defaultdict(set)
    for item in instances["annotations"]:
        categories_by_image[int(item["image_id"])].add(
            category_names[int(item["category_id"])]
        )
    return {
        "file_names": file_names,
        "captions": ordered_captions,
        "categories": categories_by_image,
    }


def _nested_order(image_ids: list[int], *, seed: int) -> list[int]:
    ordered = sorted(image_ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _fuse(native: np.ndarray, dimension: np.ndarray) -> np.ndarray:
    fused = 0.3 * native + 0.7 * dimension
    return fused / np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12, None)


def _encode_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start : start + batch_size])
            features = _normalize(model.encode_text(tokens))
            batches.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(batches)


def _encode_images(
    model: torch.nn.Module,
    preprocess: Any,
    *,
    archive_path: Path,
    image_ids: list[int],
    file_names: dict[int, str],
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    started = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive, torch.inference_mode():
        for start in range(0, len(image_ids), batch_size):
            tensors = []
            for image_id in image_ids[start : start + batch_size]:
                member = f"val2017/{file_names[image_id]}"
                image = Image.open(BytesIO(archive.read(member))).convert("RGB")
                tensors.append(preprocess(image))
            features = _normalize(model.encode_image(torch.stack(tensors)))
            batches.append(features.cpu().numpy().astype(np.float32))
            completed = min(start + batch_size, len(image_ids))
            if completed == len(image_ids) or completed % 512 == 0:
                elapsed = time.monotonic() - started
                print(f"encoded images {completed}/{len(image_ids)} in {elapsed:.1f}s", flush=True)
    return np.concatenate(batches)


def _build_or_load_cache(
    args: argparse.Namespace,
    *,
    annotations: dict[str, Any],
    image_ids: list[int],
) -> dict[str, np.ndarray]:
    if args.cache.exists():
        loaded = np.load(args.cache)
        cached_ids = loaded["image_ids"].astype(np.int64)
        if set(loaded.files) == {"image_ids", *CHANNEL_WEIGHTS} and np.array_equal(
            cached_ids,
            np.asarray(image_ids, dtype=np.int64),
        ):
            print(f"reusing embedding cache {args.cache}", flush=True)
            return {key: loaded[key].astype(np.float32) for key in loaded.files}

    model, _, preprocess = mobileclip.create_model_and_transforms(
        "mobileclip_s0",
        pretrained=str(args.checkpoint),
    )
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    captions: dict[int, list[str]] = annotations["captions"]
    categories: dict[int, set[str]] = annotations["categories"]

    native = _encode_images(
        model,
        preprocess,
        archive_path=args.images,
        image_ids=image_ids,
        file_names=annotations["file_names"],
        batch_size=args.batch_size,
    )
    subject_texts = [
        "objects: " + ", ".join(sorted(categories.get(image_id, set())))
        if categories.get(image_id)
        else "objects described as " + captions[image_id][1]
        for image_id in image_ids
    ]
    scene_texts = [". ".join(captions[image_id][1:]) for image_id in image_ids]
    subject_text = _encode_texts(
        model,
        tokenizer,
        subject_texts,
        batch_size=args.batch_size,
    )
    scene_text = _encode_texts(
        model,
        tokenizer,
        scene_texts,
        batch_size=args.batch_size,
    )
    cache = {
        "image_ids": np.asarray(image_ids, dtype=np.int64),
        "native_multimodal": native,
        "subject_content": _fuse(native, subject_text),
        "scene_theme": _fuse(native, scene_text),
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **cache)
    print(f"wrote embedding cache {args.cache}", flush=True)
    return cache


def _tokens(text: str) -> list[str]:
    return [item for item in TOKEN_RE.findall(text.lower()) if item not in STOP_WORDS]


def _bm25_rankings(
    *,
    query_texts: list[str],
    document_texts: list[str],
    limit: int,
) -> list[list[int]]:
    documents = [_tokens(item) for item in document_texts]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    rankings: list[list[int]] = []
    for query_text in query_texts:
        query_terms = set(_tokens(query_text))
        scores: list[tuple[float, int]] = []
        for index, document in enumerate(documents):
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_count = len(documents)
                frequency_count = document_frequency[term]
                inverse_frequency = math.log(
                    1 + (document_count - frequency_count + 0.5) / (frequency_count + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1 - 0.75 + 0.75 * len(document) / max(average_length, 1e-12)
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda item: (-item[0], item[1]))
        rankings.append([index for _, index in scores[:limit]])
    return rankings


def _top_rankings(similarities: np.ndarray, *, limit: int) -> list[list[int]]:
    rankings: list[list[int]] = []
    for row in similarities:
        actual_limit = min(limit, len(row))
        if actual_limit == len(row):
            candidates = np.arange(len(row))
        else:
            candidates = np.argpartition(row, -actual_limit)[-actual_limit:]
        ordered = candidates[np.lexsort((candidates, -row[candidates]))]
        rankings.append([int(item) for item in ordered])
    return rankings


def _fuse_rankings(
    rankings: dict[str, list[int]],
    *,
    rrf_k: int,
    candidate_cap: int,
    include_text: bool,
) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for channel, weight in CHANNEL_WEIGHTS.items():
        for rank, index in enumerate(rankings[channel], start=1):
            scores[index] += weight / (rrf_k + rank)
    if include_text:
        for rank, index in enumerate(rankings["local_text"], start=1):
            scores[index] += 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda index: (-scores[index], index))[:candidate_cap]


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _metrics(ranks: list[int | None], *, candidate_cap: int) -> dict[str, float]:
    count = len(ranks)
    metrics: dict[str, float] = {}
    for cutoff in (1, 3, 5, 10, 20, 50, 100):
        metrics[f"recall_at_{cutoff}"] = (
            sum(rank is not None and rank <= cutoff for rank in ranks) / count
        )
    metrics["hit_at_1"] = metrics["recall_at_1"]
    for cutoff in (5, 10, 20, 100):
        metrics[f"mrr_at_{cutoff}"] = (
            sum(1 / rank for rank in ranks if rank is not None and rank <= cutoff) / count
        )
        metrics[f"ndcg_at_{cutoff}"] = (
            sum(
                1 / math.log2(rank + 1)
                for rank in ranks
                if rank is not None and rank <= cutoff
            )
            / count
        )
    observed = [rank for rank in ranks if rank is not None]
    penalized = [rank if rank is not None else candidate_cap + 1 for rank in ranks]
    metrics.update(
        {
            "candidate_recall": len(observed) / count,
            "missed_candidate_count": float(count - len(observed)),
            "mean_observed_relevant_rank": (
                sum(observed) / len(observed) if observed else float(candidate_cap + 1)
            ),
            "mean_relevant_rank": sum(penalized) / count,
            "median_relevant_rank": _percentile(penalized, 50),
            "p75_relevant_rank": _percentile(penalized, 75),
            "p90_relevant_rank": _percentile(penalized, 90),
        }
    )
    return metrics


def _evaluate_size(
    args: argparse.Namespace,
    *,
    size: int,
    cache: dict[str, np.ndarray],
    query_vectors: np.ndarray,
    query_ids: list[int],
    query_texts: list[str],
    document_texts: list[str],
) -> dict[str, Any]:
    corpus_ids = cache["image_ids"][:size].astype(np.int64).tolist()
    corpus_position = {image_id: index for index, image_id in enumerate(corpus_ids)}
    channel_rankings = {
        channel: _top_rankings(
            query_vectors @ cache[channel][:size].T,
            limit=args.channel_limit,
        )
        for channel in CHANNEL_WEIGHTS
    }
    channel_rankings["local_text"] = _bm25_rankings(
        query_texts=query_texts,
        document_texts=document_texts[:size],
        limit=args.channel_limit,
    )

    report: dict[str, Any] = {
        "channel_relevant_recall": {},
        "vector_only": {},
        "vector_plus_text": {},
        "per_query_relevant_rank": {
            "vector_only": {},
            "vector_plus_text": {},
        },
    }
    for channel, rankings in channel_rankings.items():
        report["channel_relevant_recall"][channel] = sum(
            corpus_position[query_id] in rankings[query_index]
            for query_index, query_id in enumerate(query_ids)
        ) / len(query_ids)

    for mode, include_text in (("vector_only", False), ("vector_plus_text", True)):
        for rrf_k in args.k:
            relevant_ranks: list[int | None] = []
            for query_index, query_id in enumerate(query_ids):
                per_query = {
                    channel: rankings[query_index]
                    for channel, rankings in channel_rankings.items()
                }
                fused = _fuse_rankings(
                    per_query,
                    rrf_k=rrf_k,
                    candidate_cap=args.candidate_cap,
                    include_text=include_text,
                )
                relevant_index = corpus_position[query_id]
                try:
                    relevant_ranks.append(fused.index(relevant_index) + 1)
                except ValueError:
                    relevant_ranks.append(None)
            report[mode][str(rrf_k)] = _metrics(
                relevant_ranks,
                candidate_cap=args.candidate_cap,
            )
            report["per_query_relevant_rank"][mode][str(rrf_k)] = relevant_ranks
    return report


def main() -> None:
    args = _parse_args()
    annotations = _load_annotations(args.annotations)
    maximum_size = max(args.sizes)
    if maximum_size > len(annotations["file_names"]):
        raise ValueError("requested corpus size exceeds COCO val2017")
    if args.query_count > min(args.sizes):
        raise ValueError("query count must not exceed the smallest corpus")

    image_ids = _nested_order(list(annotations["file_names"]), seed=args.seed)[:maximum_size]
    cache = _build_or_load_cache(args, annotations=annotations, image_ids=image_ids)
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
        "mobileclip_s0",
        pretrained=str(args.checkpoint),
    )
    model.eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    query_vectors = _encode_texts(
        model,
        tokenizer,
        query_texts,
        batch_size=args.batch_size,
    )

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
        vector_best = max(
            args.k,
            key=lambda k: scales[str(size)]["vector_only"][str(k)]["ndcg_at_10"],
        )
        text_best = max(
            args.k,
            key=lambda k: scales[str(size)]["vector_plus_text"][str(k)]["ndcg_at_10"],
        )
        print(
            f"size={size}: best vector K={vector_best}, best vector+text K={text_best}",
            flush=True,
        )

    report = {
        "dataset": "COCO val2017",
        "embedding_model": "MobileCLIP-S0 (local checkpoint)",
        "seed": args.seed,
        "query_count": args.query_count,
        "query_image_ids": query_ids,
        "query_texts": query_texts,
        "sizes": args.sizes,
        "rrf_k_values": args.k,
        "channel_limit": args.channel_limit,
        "candidate_cap": args.candidate_cap,
        "dimension_weights": CHANNEL_WEIGHTS,
        "dimension_vector_fusion": {"native": 0.3, "dimension_text": 0.7},
        "scales": scales,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote report {args.output}", flush=True)


if __name__ == "__main__":
    main()
