"""Evaluate several weighted-RRF K values from one stable set of recalls."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from capsule.config import Settings
from capsule.db.repositories import EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import AssetType, EmbeddingType
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.search.fusion import FusionEngine
from capsule.search.models import FusionMethod, QueryType, SearchFilters, SearchRequest
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.search.service import SearchService
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.milvus import MilvusVectorStore

DEFAULT_DIMENSIONS = [
    EmbeddingType.NATIVE_MULTIMODAL,
    EmbeddingType.SUBJECT_CONTENT,
    EmbeddingType.SCENE_THEME,
    EmbeddingType.VISUAL_PRESENTATION,
]


@dataclass(slots=True, frozen=True)
class Case:
    case_id: str
    query: str
    relevant_asset_ids: frozenset[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/benchmarks/rrf-k-eval-50-2026-08-13.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/rrf-k-eval-results-2026-08-13.json"),
    )
    parser.add_argument(
        "--workspace",
        default="workspace_01KZR5S708GN98A9FH4MC0MPN5",
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 30, 60, 90, 120])
    parser.add_argument("--concurrency", type=int, default=3)
    return parser.parse_args()


def _load_cases(path: Path) -> list[Case]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Case(
            case_id=item["id"],
            query=item["query"],
            relevant_asset_ids=frozenset(item["relevant_asset_ids"]),
        )
        for item in raw
    ]


def _reciprocal_rank(ranked_ids: list[str], relevant: frozenset[str], cutoff: int) -> float:
    for rank, asset_id in enumerate(ranked_ids[:cutoff], start=1):
        if asset_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked_ids: list[str], relevant: frozenset[str], cutoff: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, asset_id in enumerate(ranked_ids[:cutoff], start=1)
        if asset_id in relevant
    )
    ideal_count = min(len(relevant), cutoff)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def _case_metrics(ranked_ids: list[str], relevant: frozenset[str]) -> dict[str, float]:
    return {
        "hit_at_1": float(bool(ranked_ids) and ranked_ids[0] in relevant),
        "precision_at_5": len(set(ranked_ids[:5]) & relevant) / 5,
        "recall_at_5": len(set(ranked_ids[:5]) & relevant) / len(relevant),
        "recall_at_10": len(set(ranked_ids[:10]) & relevant) / len(relevant),
        "mrr_at_10": _reciprocal_rank(ranked_ids, relevant, 10),
        "ndcg_at_10": _ndcg(ranked_ids, relevant, 10),
    }


async def _collect_case(
    case: Case,
    *,
    workspace_id: str,
    service: SearchService,
    repository: PostgresAssetSearchRepository,
    k_values: list[int],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        request = SearchRequest(
            workspace_id=workspace_id,
            query_type=QueryType.TEXT,
            query_text=case.query,
            embedding_types=DEFAULT_DIMENSIONS,
            fusion_method=FusionMethod.WEIGHTED_RRF,
            filters=SearchFilters(asset_type=[AssetType.IMAGE]),
            top_k=20,
        )
        parsed, parser_reasons = await service._query_parser.parse(request, image_url=None)
        plan = await service._query_embedding.embed(request, parsed, image_url=None)
        plan = await service._prepare_search_vector_indexes(
            plan=plan,
            workspace_id=workspace_id,
        )
        recalled = await service._recall.search(
            plan=plan,
            workspace_id=workspace_id,
            filters=request.filters,
            top_k=request.top_k,
        )
        text_hits = await repository.search_text(
            workspace_id=workspace_id,
            query_text=case.query,
            filters=request.filters,
            created_by=request.created_by,
            limit=60,
        )

    rankings: dict[str, list[str]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for k in k_values:
        fused = FusionEngine(rrf_k=k, candidate_cap=300).fuse(
            recalled.channels,
            FusionMethod.WEIGHTED_RRF,
            text_hits=text_hits,
        )
        ranked_ids = [item.asset_id for item in fused]
        rankings[str(k)] = ranked_ids[:10]
        metrics[str(k)] = _case_metrics(ranked_ids, case.relevant_asset_ids)

    return {
        "id": case.case_id,
        "query": case.query,
        "relevant_asset_ids": sorted(case.relevant_asset_ids),
        "dimension_queries": [item.model_dump(mode="json") for item in parsed.dimension_queries],
        "effective_weights": {
            item.embedding_type.value: item.weight for item in plan.vectors
        },
        "channel_hit_counts": {
            item.query_vector.embedding_type.value: len(item.hits) for item in recalled.channels
        },
        "text_hit_count": len(text_hits),
        "degraded_reasons": list(
            dict.fromkeys((*parser_reasons, *plan.degraded_reasons, *recalled.degraded_reasons))
        ),
        "rankings": rankings,
        "metrics": metrics,
    }


def _aggregate(cases: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    metric_names = [
        "hit_at_1",
        "precision_at_5",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    ]
    reference_key = "60" if 60 in k_values else str(k_values[-1])
    aggregate: dict[str, Any] = {}
    for k in k_values:
        key = str(k)
        row = {
            metric: mean(case["metrics"][key][metric] for case in cases)
            for metric in metric_names
        }
        row["changed_top_10_vs_reference"] = sum(
            case["rankings"][key] != case["rankings"][reference_key] for case in cases
        )
        row["changed_top_1_vs_reference"] = sum(
            case["rankings"][key][:1] != case["rankings"][reference_key][:1]
            for case in cases
        )
        aggregate[key] = row
    return aggregate


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    database = Database(settings)
    model_client = DoubaoClient(settings)
    vectors = MilvusVectorStore(settings)
    repository = PostgresAssetSearchRepository(database)
    embedding_service = AssetEmbeddingService(
        settings=settings,
        repository=EmbeddingRepository(database),
        model_client=model_client,
        vector_store=vectors,
        artifact_reader=ObjectStorage(settings),
    )
    service = SearchService(
        query_embedding=QueryEmbeddingService(model_client, settings),
        query_parser=QueryParser(model_client),
        recall=MultiChannelRecall(vectors, settings),
        assets=repository,
        text_recall=repository,
        search_vector_preparer=embedding_service,
        settings=settings,
    )
    cases = _load_cases(args.cases)
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        results = await asyncio.gather(
            *(
                _collect_case(
                    case,
                    workspace_id=args.workspace,
                    service=service,
                    repository=repository,
                    k_values=args.k,
                    semaphore=semaphore,
                )
                for case in cases
            )
        )
    finally:
        await model_client.close()
        await database.dispose()

    weight_summary = {
        embedding_type.value: mean(
            case["effective_weights"].get(embedding_type.value, 0.0) for case in results
        )
        for embedding_type in DEFAULT_DIMENSIONS
    }
    return {
        "workspace_id": args.workspace,
        "dataset": str(args.cases),
        "case_count": len(results),
        "k_values": args.k,
        "method": "weighted_rrf",
        "recall_reused_across_k": True,
        "dimensions": [item.value for item in DEFAULT_DIMENSIONS],
        "mean_effective_weights": weight_summary,
        "aggregate": _aggregate(results, args.k),
        "cases": results,
    }


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), **report["aggregate"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
