import asyncio
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from capsule.search.models import QueryType, SearchRequest, SearchResponse


class EvaluationCase(BaseModel):
    request: SearchRequest
    relevant_asset_ids: set[str] = Field(min_length=1)


class QueryMetrics(BaseModel):
    query_type: QueryType
    precision_at_5: float
    recall_at_10: float


class EvaluationReport(BaseModel):
    case_count: int
    precision_at_5: float
    recall_at_10: float
    by_query_type: dict[str, dict[str, float]]
    thresholds: dict[str, float]
    passed: bool


async def evaluate_search_file(
    path: Path,
    *,
    api_base_url: str,
    concurrency: int = 4,
) -> EvaluationReport:
    cases = _load_cases(path)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=api_base_url.rstrip("/"), timeout=300) as client:
        metrics = await asyncio.gather(*(_evaluate_case(client, semaphore, item) for item in cases))
    return _report(metrics)


def _load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid evaluation JSONL at line {line_number}") from exc
    if not cases:
        raise ValueError("evaluation dataset is empty")
    return cases


async def _evaluate_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: EvaluationCase,
) -> QueryMetrics:
    async with semaphore:
        response = await client.post(
            "/api/v1/search",
            json=case.request.model_dump(mode="json"),
        )
        response.raise_for_status()
    result = SearchResponse.model_validate(response.json())
    ranked_ids = [item.asset_id for item in result.results]
    relevant = case.relevant_asset_ids
    return QueryMetrics(
        query_type=case.request.query_type,
        precision_at_5=len(set(ranked_ids[:5]) & relevant) / 5,
        recall_at_10=len(set(ranked_ids[:10]) & relevant) / len(relevant),
    )


def _report(metrics: list[QueryMetrics]) -> EvaluationReport:
    grouped: dict[QueryType, list[QueryMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.query_type, []).append(item)
    by_type: dict[str, dict[str, float]] = {
        query_type.value: {
            "precision_at_5": _average(item.precision_at_5 for item in items),
            "recall_at_10": _average(item.recall_at_10 for item in items),
        }
        for query_type, items in grouped.items()
    }
    thresholds = {
        QueryType.TEXT.value: 0.75,
        QueryType.IMAGE.value: 0.70,
        QueryType.IMAGE_TEXT.value: 0.70,
        "recall_at_10": 0.75,
    }
    passed = (
        all(
            by_type.get(query_type, {}).get("precision_at_5", 0) >= threshold
            for query_type, threshold in thresholds.items()
            if query_type != "recall_at_10" and query_type in by_type
        )
        and _average(item.recall_at_10 for item in metrics) >= thresholds["recall_at_10"]
    )
    return EvaluationReport(
        case_count=len(metrics),
        precision_at_5=_average(item.precision_at_5 for item in metrics),
        recall_at_10=_average(item.recall_at_10 for item in metrics),
        by_query_type=by_type,
        thresholds=thresholds,
        passed=passed,
    )


def _average(values: Any) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0
