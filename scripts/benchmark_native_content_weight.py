"""Evaluate the native/original-content share used inside dimension vectors.

This benchmark rebuilds both corpus-side and query-side fused vectors from the
raw native and dimension embeddings.  Weighted-RRF K and route weights remain
fixed so the result isolates the vector-fusion ratio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from capsule.config import Settings
from capsule.db.repositories import EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import AssetType, EmbeddingType
from capsule.model_clients.doubao import DoubaoClient
from capsule.search.models import SearchFilters
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.vectorstore.milvus import MilvusVectorStore

DIMENSIONS = (
    EmbeddingType.NATIVE_MULTIMODAL,
    EmbeddingType.SUBJECT_CONTENT,
    EmbeddingType.SCENE_THEME,
    EmbeddingType.VISUAL_PRESENTATION,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmarks/rrf-k-eval-results-2026-08-13.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/benchmarks/native-content-weight-real-50-2026-08-13.json"
        ),
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=Path(
            "data/benchmarks/native-content-weight-query-cache-2026-08-13.npz"
        ),
    )
    parser.add_argument(
        "--workspace",
        default="workspace_01KZR5S708GN98A9FH4MC0MPN5",
    )
    parser.add_argument(
        "--native-weights",
        type=float,
        nargs="+",
        default=[item / 10 for item in range(11)],
    )
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--channel-limit", type=int, default=60)
    parser.add_argument("--candidate-cap", type=int, default=300)
    parser.add_argument("--embedding-concurrency", type=int, default=3)
    return parser.parse_args()


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("vectors must be finite and non-zero")
    return vectors / norms


def _fuse_rows(native: np.ndarray, dimension: np.ndarray, weight: float) -> np.ndarray:
    fused = weight * _normalize_rows(native) + (1.0 - weight) * _normalize_rows(dimension)
    return _normalize_rows(fused)


def _rank(similarities: np.ndarray, asset_ids: list[str], limit: int) -> list[str]:
    ranked = sorted(
        range(len(asset_ids)),
        key=lambda index: (-float(similarities[index]), asset_ids[index]),
    )
    return [asset_ids[index] for index in ranked[:limit]]


def _rrf(
    rankings: dict[str, list[str]],
    route_weights: dict[str, float],
    *,
    rrf_k: int,
    candidate_cap: int,
) -> list[str]:
    scores: dict[str, float] = {}
    for route, ranking in rankings.items():
        weight = 1.0 if route == "local_text" else route_weights[route]
        for rank, asset_id in enumerate(ranking, start=1):
            scores[asset_id] = scores.get(asset_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores, key=lambda asset_id: (-scores[asset_id], asset_id))[:candidate_cap]


def _reciprocal_rank(ranking: list[str], relevant: set[str], cutoff: int) -> float:
    for rank, asset_id in enumerate(ranking[:cutoff], start=1):
        if asset_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(ranking: list[str], relevant: set[str], cutoff: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, asset_id in enumerate(ranking[:cutoff], start=1)
        if asset_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), cutoff) + 1)
    )
    return dcg / ideal if ideal else 0.0


def _metrics(ranking: list[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    first_rank = next(
        (rank for rank, asset_id in enumerate(ranking, start=1) if asset_id in relevant),
        len(ranking) + 1,
    )
    for cutoff in (1, 3, 5, 10, 20):
        found = len(set(ranking[:cutoff]) & relevant)
        result[f"recall_at_{cutoff}"] = found / len(relevant)
        result[f"precision_at_{cutoff}"] = found / cutoff
        result[f"mrr_at_{cutoff}"] = _reciprocal_rank(ranking, relevant, cutoff)
        result[f"ndcg_at_{cutoff}"] = _ndcg(ranking, relevant, cutoff)
    result["hit_at_1"] = float(bool(ranking) and ranking[0] in relevant)
    result["first_relevant_rank"] = float(first_rank)
    return result


async def _load_corpus(
    *,
    workspace_id: str,
    settings: Settings,
    repository: EmbeddingRepository,
    vector_store: MilvusVectorStore,
) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    for embedding_type in DIMENSIONS:
        assets = await repository.list_indexed_cluster_embeddings(
            workspace_id=workspace_id,
            embedding_type=embedding_type.value,
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            milvus_collection=settings.milvus_collection,
        )
        assets = sorted(assets, key=lambda item: item.asset_id)
        raw = await vector_store.fetch_vectors([item.embedding_id for item in assets])
        loaded = [item for item in assets if item.embedding_id in raw]
        corpus[embedding_type.value] = {
            "asset_ids": [item.asset_id for item in loaded],
            "vectors": _normalize_rows(
                np.asarray([raw[item.embedding_id] for item in loaded], dtype=np.float32)
            ),
        }
    return corpus


async def _load_query_vectors(
    *,
    cases: list[dict[str, Any]],
    cache_path: Path,
    model_client: DoubaoClient,
    concurrency: int,
) -> dict[str, np.ndarray]:
    texts = sorted(
        {
            item["query"]
            for case in cases
            for item in case["dimension_queries"]
        }
        | {case["query"] for case in cases}
    )
    if await asyncio.to_thread(cache_path.exists):
        loaded = await asyncio.to_thread(np.load, cache_path)
        cached_texts = loaded["texts"].astype(str).tolist()
        if cached_texts == texts:
            return {
                text: vector
                for text, vector in zip(cached_texts, loaded["vectors"], strict=True)
            }

    semaphore = asyncio.Semaphore(concurrency)

    async def embed(text: str) -> np.ndarray:
        async with semaphore:
            result = await model_client.embed_text(text)
        vector = np.asarray(result.vector, dtype=np.float32)[None, :]
        return _normalize_rows(vector)[0]

    vectors = await asyncio.gather(*(embed(text) for text in texts))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        texts=np.asarray(texts),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    return dict(zip(texts, vectors, strict=True))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.input.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = source["cases"]
    settings = Settings()
    database = Database(settings)
    embedding_repository = EmbeddingRepository(database)
    text_repository = PostgresAssetSearchRepository(database)
    vector_store = MilvusVectorStore(settings)
    model_client = DoubaoClient(settings)
    try:
        corpus, query_vectors = await asyncio.gather(
            _load_corpus(
                workspace_id=args.workspace,
                settings=settings,
                repository=embedding_repository,
                vector_store=vector_store,
            ),
            _load_query_vectors(
                cases=cases,
                cache_path=args.query_cache,
                model_client=model_client,
                concurrency=args.embedding_concurrency,
            ),
        )
        text_rankings = await asyncio.gather(
            *(
                text_repository.search_text(
                    workspace_id=args.workspace,
                    query_text=case["query"],
                    filters=SearchFilters(asset_type=[AssetType.IMAGE]),
                    created_by="user_demo",
                    limit=args.channel_limit,
                )
                for case in cases
            )
        )
    finally:
        await model_client.close()
        await database.dispose()

    native = corpus[EmbeddingType.NATIVE_MULTIMODAL.value]
    native_by_asset = {
        asset_id: vector
        for asset_id, vector in zip(native["asset_ids"], native["vectors"], strict=True)
    }
    reports: dict[str, Any] = {}
    metric_names: set[str] = set()
    for native_weight in args.native_weights:
        if not 0.0 <= native_weight <= 1.0:
            raise ValueError("native weights must be between 0 and 1")
        key = f"{native_weight:.1f}"
        case_reports: list[dict[str, Any]] = []
        for case, text_hits in zip(cases, text_rankings, strict=True):
            original_query = query_vectors[case["query"]][None, :]
            dimension_queries = {
                item["embedding_type"]: query_vectors[item["query"]][None, :]
                for item in case["dimension_queries"]
            }
            rankings: dict[str, list[str]] = {}
            for embedding_type in DIMENSIONS:
                route = embedding_type.value
                route_corpus = corpus[route]
                if embedding_type is EmbeddingType.NATIVE_MULTIMODAL:
                    corpus_vectors = route_corpus["vectors"]
                    query = original_query
                else:
                    eligible = [
                        index
                        for index, asset_id in enumerate(route_corpus["asset_ids"])
                        if asset_id in native_by_asset
                    ]
                    route_asset_ids = [route_corpus["asset_ids"][index] for index in eligible]
                    dimension_vectors = route_corpus["vectors"][eligible]
                    native_vectors = np.asarray(
                        [native_by_asset[asset_id] for asset_id in route_asset_ids],
                        dtype=np.float32,
                    )
                    corpus_vectors = _fuse_rows(
                        native_vectors,
                        dimension_vectors,
                        native_weight,
                    )
                    query = _fuse_rows(
                        original_query,
                        dimension_queries[route],
                        native_weight,
                    )
                    route_corpus = {"asset_ids": route_asset_ids}
                rankings[route] = _rank(
                    (query @ corpus_vectors.T)[0],
                    route_corpus["asset_ids"],
                    args.channel_limit,
                )
            rankings["local_text"] = [item.asset_id for item in text_hits]
            route_weights = {
                route: float(weight) for route, weight in case["effective_weights"].items()
            }
            fused = _rrf(
                rankings,
                route_weights,
                rrf_k=args.rrf_k,
                candidate_cap=args.candidate_cap,
            )
            case_metrics = _metrics(fused, set(case["relevant_asset_ids"]))
            metric_names.update(case_metrics)
            case_reports.append(
                {
                    "id": case["id"],
                    "ranking_top_20": fused[:20],
                    "metrics": case_metrics,
                }
            )
        reports[key] = {
            "aggregate": {
                metric: mean(item["metrics"][metric] for item in case_reports)
                for metric in sorted(metric_names)
            },
            "cases": case_reports,
        }

    return {
        "dataset": str(args.input),
        "workspace_id": args.workspace,
        "case_count": len(cases),
        "corpus_counts": {
            route: len(item["asset_ids"]) for route, item in corpus.items()
        },
        "native_content_weights": args.native_weights,
        "rrf_k": args.rrf_k,
        "channel_limit": args.channel_limit,
        "candidate_cap": args.candidate_cap,
        "fusion_scope": "both corpus and query vectors",
        "route_weights": "reuse each case's production query-enhancement weights",
        "results": reports,
    }


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        weight: result["aggregate"] for weight, result in report["results"].items()
    }
    print(json.dumps({"output": str(args.output), "results": summary}, indent=2))


if __name__ == "__main__":
    main()
