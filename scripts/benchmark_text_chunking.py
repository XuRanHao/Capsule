"""Benchmark text chunk sizes with real queries and evidence labels.

The benchmark keeps every chunk-size variant in its own Capsule workspace,
indexes only the native text channel, and evaluates at the source-document
level so different chunk IDs remain directly comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any
from zipfile import ZipFile

from capsule.bootstrap import bootstrap_runtime
from capsule.config import Settings
from capsule.db.repositories import EmbeddingRepository
from capsule.db.session import Database
from capsule.enums import AssetType, EmbeddingType
from capsule.model_clients.doubao import DoubaoClient
from capsule.pipeline.embedding import AssetEmbeddingService
from capsule.pipeline.runner import PipelineRunner
from capsule.schemas import EmbeddingResult
from capsule.search.models import SearchFilters, SearchRequest, SearchResult
from capsule.search.query_embedding import QueryEmbeddingService
from capsule.search.query_parser import QueryParser
from capsule.search.recall import MultiChannelRecall
from capsule.search.repositories import PostgresAssetSearchRepository
from capsule.search.service import SearchService
from capsule.storage.object_storage import ObjectStorage
from capsule.vectorstore.milvus import MilvusVectorStore

DEFAULT_TARGETS = (300, 400, 500, 600, 700, 800)
DEFAULT_CORPUS_SIZE = 1_000
DEFAULT_QUERY_CONCURRENCY = 8
RECALL_CUTOFFS = (5, 10, 20)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(slots=True, frozen=True)
class BenchmarkQuery:
    query_id: str
    text: str
    relevant_doc_ids: frozenset[str]
    evidence: Mapping[str, tuple[tuple[str, ...], ...]]


@dataclass(slots=True, frozen=True)
class PreparedDataset:
    name: str
    workspace_prefix: str
    queries: tuple[BenchmarkQuery, ...]
    corpus_dir: Path
    selected_doc_ids: tuple[str, ...]
    selection_sha256: str


@dataclass(slots=True, frozen=True)
class ChunkProfile:
    target_tokens: int
    min_tokens: int
    max_tokens: int
    merge_max_tokens: int
    parent_max_tokens: int = 2_000


class CachedQueryEmbeddingClient:
    """Reuse identical query vectors across isolated benchmark workspaces."""

    def __init__(self, client: DoubaoClient) -> None:
        self._client = client
        self._text_tasks: dict[str, asyncio.Task[EmbeddingResult]] = {}

    async def embed_text(self, text: str) -> EmbeddingResult:
        task = self._text_tasks.get(text)
        if task is None:
            task = asyncio.create_task(self._client.embed_text(text))
            self._text_tasks[text] = task
        try:
            return await task
        except Exception:
            self._text_tasks.pop(text, None)
            raise

    async def embed_image(self, image_url: str) -> EmbeddingResult:
        return await self._client.embed_image(image_url)

    async def embed_image_text(self, image_url: str, text: str) -> EmbeddingResult:
        return await self._client.embed_image_text(image_url, text)


def _profile(target_tokens: int) -> ChunkProfile:
    if target_tokens < 1:
        raise ValueError("target token count must be positive")
    scale = target_tokens / 400
    return ChunkProfile(
        target_tokens=target_tokens,
        min_tokens=max(1, round(250 * scale)),
        max_tokens=max(target_tokens, round(500 * scale)),
        merge_max_tokens=max(target_tokens, round(600 * scale)),
    )


def _settings_for_profile(base: Settings, profile: ChunkProfile) -> Settings:
    return base.model_copy(
        update={
            "document_chunk_min_tokens": profile.min_tokens,
            "document_chunk_target_tokens": profile.target_tokens,
            "document_chunk_max_tokens": profile.max_tokens,
            "document_chunk_merge_max_tokens": profile.merge_max_tokens,
            "document_parent_max_tokens": profile.parent_max_tokens,
            "assetization_version": f"text-chunk-eval-v1-{profile.target_tokens}",
        }
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_beir(beir_zip: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    with ZipFile(beir_zip) as archive:
        queries = {
            str(row["_id"]): str(row["text"])
            for row in (
                json.loads(line)
                for line in archive.read("scifact/queries.jsonl").decode().splitlines()
            )
        }
        qrels: dict[str, set[str]] = defaultdict(set)
        rows = csv.DictReader(
            archive.read("scifact/qrels/test.tsv").decode().splitlines(),
            delimiter="\t",
        )
        for row in rows:
            if int(row["score"]) > 0:
                qrels[str(row["query-id"])].add(str(row["corpus-id"]))
    return queries, qrels


def _terms(text: str) -> frozenset[str]:
    return frozenset(TOKEN_PATTERN.findall(text.lower()))


def _select_corpus(
    corpus: Mapping[str, dict[str, Any]],
    *,
    queries: Sequence[BenchmarkQuery],
    corpus_size: int,
) -> tuple[str, ...]:
    mandatory = {doc_id for query in queries for doc_id in query.relevant_doc_ids}
    if corpus_size < len(mandatory):
        raise ValueError(
            f"corpus_size={corpus_size} is smaller than {len(mandatory)} gold documents"
        )
    document_terms = {
        doc_id: _terms(f"{row.get('title', '')} {' '.join(row.get('abstract', []))}")
        for doc_id, row in corpus.items()
    }
    document_frequency = Counter(term for terms in document_terms.values() for term in terms)
    query_terms = [_terms(query.text) for query in queries]
    total_documents = len(corpus)

    def hard_negative_score(doc_id: str) -> tuple[float, str]:
        terms = document_terms[doc_id]
        score = max(
            (
                sum(
                    math.log((total_documents + 1) / (document_frequency[term] + 1))
                    for term in terms.intersection(candidate)
                )
                / math.sqrt(max(1, len(candidate)))
                for candidate in query_terms
            ),
            default=0.0,
        )
        return score, doc_id

    candidates = sorted(
        (doc_id for doc_id in corpus if doc_id not in mandatory),
        key=hard_negative_score,
        reverse=True,
    )
    selected = sorted(mandatory) + candidates[: corpus_size - len(mandatory)]
    return tuple(sorted(selected))


def prepare_dataset(
    *,
    beir_zip: Path,
    official_root: Path,
    output_root: Path,
    corpus_size: int,
) -> PreparedDataset:
    official_corpus = {
        str(row["doc_id"]): row for row in _load_jsonl(official_root / "corpus.jsonl")
    }
    claims = {str(row["id"]): row for row in _load_jsonl(official_root / "claims_dev.jsonl")}
    beir_queries, qrels = _load_beir(beir_zip)
    benchmark_queries: list[BenchmarkQuery] = []
    for query_id in sorted(qrels, key=int):
        claim = claims[query_id]
        evidence: dict[str, tuple[tuple[str, ...], ...]] = {}
        for doc_id, rationales in claim.get("evidence", {}).items():
            abstract = official_corpus[str(doc_id)]["abstract"]
            evidence[str(doc_id)] = tuple(
                tuple(str(abstract[index]) for index in rationale["sentences"])
                for rationale in rationales
            )
        benchmark_queries.append(
            BenchmarkQuery(
                query_id=query_id,
                text=beir_queries[query_id],
                relevant_doc_ids=frozenset(qrels[query_id]),
                evidence=evidence,
            )
        )

    selected_doc_ids = _select_corpus(
        official_corpus,
        queries=benchmark_queries,
        corpus_size=corpus_size,
    )
    selection_sha256 = hashlib.sha256("\n".join(selected_doc_ids).encode("utf-8")).hexdigest()
    corpus_dir = output_root / f"corpus-{corpus_size}-{selection_sha256[:12]}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id in selected_doc_ids:
        row = official_corpus[doc_id]
        paragraphs = "\n\n".join(str(sentence).strip() for sentence in row["abstract"])
        content = f"# {str(row['title']).strip()}\n\n## Abstract\n\n{paragraphs}\n"
        path = corpus_dir / f"{doc_id}.md"
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
    manifest = {
        "dataset": "SciFact dev / BEIR SciFact test qrels",
        "query_count": len(benchmark_queries),
        "evidence_query_count": sum(bool(query.evidence) for query in benchmark_queries),
        "corpus_size": len(selected_doc_ids),
        "selection_sha256": selection_sha256,
        "selected_doc_ids": selected_doc_ids,
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PreparedDataset(
        name="SciFact dev / BEIR SciFact",
        workspace_prefix="text_chunk_eval",
        queries=tuple(benchmark_queries),
        corpus_dir=corpus_dir,
        selected_doc_ids=selected_doc_ids,
        selection_sha256=selection_sha256,
    )


def _normalized_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def _answer_evidence(context: str, answers: Sequence[str]) -> tuple[str, ...]:
    sentences = [
        match.group(0).strip()
        for match in re.finditer(r"[^。！？!?；;\n]+[。！？!?；;]?", context)
        if match.group(0).strip()
    ]
    for answer in answers:
        normalized_answer = _normalized_match_text(answer)
        if not normalized_answer:
            continue
        for index, sentence in enumerate(sentences):
            if normalized_answer not in _normalized_match_text(sentence):
                continue
            start = max(0, index - 1)
            end = min(len(sentences), index + 2)
            return tuple(sentences[start:end])
    return ()


def _stratified_longbench_sample(
    candidates: Sequence[tuple[dict[str, Any], tuple[str, ...], float]],
    *,
    count: int,
) -> list[tuple[dict[str, Any], tuple[str, ...], float]]:
    buckets: dict[int, list[tuple[dict[str, Any], tuple[str, ...], float]]] = defaultdict(list)
    for candidate in candidates:
        bucket = min(9, int(candidate[2] * 10))
        buckets[bucket].append(candidate)
    for values in buckets.values():
        values.sort(key=lambda item: hashlib.sha256(str(item[0]["_id"]).encode()).hexdigest())
    selected: list[tuple[dict[str, Any], tuple[str, ...], float]] = []
    while len(selected) < count and any(buckets.values()):
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < count:
                selected.append(buckets[bucket].pop(0))
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} answer-locatable LongBench rows are available")
    return selected


def prepare_longbench_dataset(
    *,
    longbench_root: Path,
    output_root: Path,
    per_task: int,
) -> PreparedDataset:
    selected_rows: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []
    for task_name in ("multifieldqa_zh", "dureader"):
        candidates: list[tuple[dict[str, Any], tuple[str, ...], float]] = []
        for row in _load_jsonl(longbench_root / f"{task_name}.jsonl"):
            evidence = _answer_evidence(row["context"], row["answers"])
            if not evidence:
                continue
            normalized_context = _normalized_match_text(row["context"])
            normalized_evidence = _normalized_match_text(evidence[len(evidence) // 2])
            offset = normalized_context.find(normalized_evidence)
            position = max(0, offset) / max(1, len(normalized_context))
            candidates.append((row, evidence, position))
        selected_rows.extend(
            (task_name, row, evidence)
            for row, evidence, _ in _stratified_longbench_sample(
                candidates,
                count=per_task,
            )
        )

    selected_rows.sort(key=lambda item: (item[0], str(item[1]["_id"])))
    selected_doc_ids = tuple(
        f"{task_name}_{index:03d}" for index, (task_name, _, _) in enumerate(selected_rows)
    )
    selection_payload = [
        f"{doc_id}:{row['_id']}"
        for doc_id, (_, row, _) in zip(selected_doc_ids, selected_rows, strict=True)
    ]
    selection_sha256 = hashlib.sha256("\n".join(selection_payload).encode()).hexdigest()
    corpus_dir = output_root / f"corpus-{len(selected_rows)}-{selection_sha256[:12]}"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    queries: list[BenchmarkQuery] = []
    for doc_id, (task_name, row, evidence) in zip(
        selected_doc_ids,
        selected_rows,
        strict=True,
    ):
        content = f"# LongBench {task_name}\n\n{row['context'].strip()}\n"
        path = corpus_dir / f"{doc_id}.md"
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        queries.append(
            BenchmarkQuery(
                query_id=str(row["_id"]),
                text=str(row["input"]),
                relevant_doc_ids=frozenset({doc_id}),
                evidence={doc_id: (evidence,)},
            )
        )
    manifest = {
        "dataset": "LongBench MultiFieldQA-zh + DuReader",
        "query_count": len(queries),
        "corpus_size": len(selected_doc_ids),
        "per_task": per_task,
        "selection_sha256": selection_sha256,
        "selection": selection_payload,
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PreparedDataset(
        name="LongBench MultiFieldQA-zh + DuReader",
        workspace_prefix="long_text_chunk_eval",
        queries=tuple(queries),
        corpus_dir=corpus_dir,
        selected_doc_ids=selected_doc_ids,
        selection_sha256=selection_sha256,
    )


def _source_doc_id(result: SearchResult) -> str | None:
    if result.source_file is None:
        return None
    return Path(result.source_file.original_file_name).stem


def _evidence_coverage(query: BenchmarkQuery, results: Sequence[SearchResult]) -> float:
    contexts_by_doc: dict[str, list[str]] = defaultdict(list)
    for result in results:
        doc_id = _source_doc_id(result)
        if doc_id is not None and result.raw_content:
            contexts_by_doc[doc_id].append(result.raw_content)
    best = 0.0
    for doc_id, rationales in query.evidence.items():
        for rationale in rationales:
            for context in contexts_by_doc.get(doc_id, []):
                present = sum(sentence in context for sentence in rationale)
                best = max(best, present / len(rationale))
    return best


def score_query(query: BenchmarkQuery, results: Sequence[SearchResult]) -> dict[str, Any]:
    ranked_doc_ids = [doc_id for result in results if (doc_id := _source_doc_id(result))]
    evidence_coverage = _evidence_coverage(query, results) if query.evidence else None
    row = {
        "query_id": query.query_id,
        "query": query.text,
        "relevant_doc_ids": sorted(query.relevant_doc_ids),
        "ranked_doc_ids": ranked_doc_ids,
        "evidence_sentence_coverage_at_20": evidence_coverage,
        "complete_evidence_at_20": (
            evidence_coverage == 1.0 if evidence_coverage is not None else None
        ),
    }
    row.update(_ranking_metrics(query.relevant_doc_ids, ranked_doc_ids))
    return row


def _ranking_metrics(
    relevant_doc_ids: Sequence[str] | frozenset[str],
    ranked_doc_ids: Sequence[str],
) -> dict[str, float]:
    relevant = set(relevant_doc_ids)
    metrics: dict[str, float] = {}
    for cutoff in RECALL_CUTOFFS:
        candidates = ranked_doc_ids[:cutoff]
        unique_candidates = set(candidates)
        metrics[f"recall_at_{cutoff}"] = len(unique_candidates & relevant) / len(relevant)
        metrics[f"duplicate_rate_at_{cutoff}"] = (
            1 - len(unique_candidates) / len(candidates) if candidates else 0.0
        )
    metrics["reciprocal_rank_at_20"] = next(
        (
            1.0 / rank
            for rank, doc_id in enumerate(ranked_doc_ids[:20], start=1)
            if doc_id in relevant
        ),
        0.0,
    )
    return metrics


def _upgrade_query_row(row: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = dict(row)
    upgraded.update(
        _ranking_metrics(
            row["relevant_doc_ids"],
            row["ranked_doc_ids"],
        )
    )
    return upgraded


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    evidence_rows = [row for row in rows if row["evidence_sentence_coverage_at_20"] is not None]
    return {
        "query_count": len(rows),
        "evidence_query_count": len(evidence_rows),
        "recall_at_5": fmean(float(row["recall_at_5"]) for row in rows),
        "recall_at_10": fmean(float(row["recall_at_10"]) for row in rows),
        "recall_at_20": fmean(float(row["recall_at_20"]) for row in rows),
        "reciprocal_rank_at_20": fmean(
            float(row["reciprocal_rank_at_20"]) for row in rows
        ),
        "duplicate_rate_at_5": fmean(float(row["duplicate_rate_at_5"]) for row in rows),
        "duplicate_rate_at_10": fmean(
            float(row["duplicate_rate_at_10"]) for row in rows
        ),
        "duplicate_rate_at_20": fmean(float(row["duplicate_rate_at_20"]) for row in rows),
        "evidence_sentence_coverage_at_20": fmean(
            float(row["evidence_sentence_coverage_at_20"]) for row in evidence_rows
        ),
        "complete_evidence_rate_at_20": fmean(
            1.0 if row["complete_evidence_at_20"] else 0.0 for row in evidence_rows
        ),
    }


async def _chunk_statistics(
    repository: EmbeddingRepository,
    *,
    workspace_id: str,
) -> dict[str, float | int]:
    assets = await repository.list_assets(workspace_id=workspace_id)
    chunks = [
        asset
        for asset in assets
        if asset.asset_type == AssetType.MARKDOWN_BLOCK.value and asset.index_role == "child"
    ]
    token_counts = [
        int(asset.file_info.get("token_count", 0))
        for asset in chunks
        if int(asset.file_info.get("token_count", 0)) > 0
    ]
    ordered = sorted(token_counts)
    return {
        "child_chunk_count": len(chunks),
        "mean_child_tokens": fmean(token_counts) if token_counts else 0.0,
        "p95_child_tokens": (
            ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)] if ordered else 0
        ),
        "oversized_child_count": sum(bool(asset.file_info.get("oversized")) for asset in chunks),
    }


async def _evaluate_workspace(
    *,
    settings: Settings,
    workspace_id: str,
    queries: Sequence[BenchmarkQuery],
    query_client: CachedQueryEmbeddingClient,
    concurrency: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    database = Database(settings)
    vectors = MilvusVectorStore(settings)
    assets = PostgresAssetSearchRepository(database)
    service = SearchService(
        query_embedding=QueryEmbeddingService(query_client, settings),
        recall=MultiChannelRecall(vectors, settings),
        assets=assets,
        settings=settings,
        query_parser=QueryParser(),
    )
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    progress_lock = asyncio.Lock()

    async def evaluate(query: BenchmarkQuery) -> dict[str, Any]:
        nonlocal completed
        async with semaphore:
            response = await service.search(
                SearchRequest(
                    workspace_id=workspace_id,
                    query_type="text",
                    query_text=query.text,
                    embedding_types=[EmbeddingType.NATIVE_MULTIMODAL],
                    filters=SearchFilters(asset_type=[AssetType.MARKDOWN_BLOCK]),
                    top_k=20,
                )
            )
        row = score_query(query, response.results)
        async with progress_lock:
            completed += 1
            if completed % 25 == 0 or completed == len(queries):
                print(
                    f"[{workspace_id}] evaluated {completed}/{len(queries)} queries",
                    flush=True,
                )
        return row

    try:
        rows = list(await asyncio.gather(*(evaluate(query) for query in queries)))
        metrics = aggregate_metrics(rows)
        metrics.update(
            await _chunk_statistics(
                EmbeddingRepository(database),
                workspace_id=workspace_id,
            )
        )
        return metrics, rows
    finally:
        await database.dispose()


async def _run_variant(
    *,
    base_settings: Settings,
    profile: ChunkProfile,
    dataset: PreparedDataset,
    model_client: DoubaoClient,
    query_client: CachedQueryEmbeddingClient,
    output_root: Path,
    query_concurrency: int,
) -> dict[str, Any]:
    workspace_id = f"{dataset.workspace_prefix}_{profile.target_tokens}"
    settings = _settings_for_profile(base_settings, profile)
    print(f"[{workspace_id}] bootstrapping workspace", flush=True)
    bootstrap = await bootstrap_runtime(
        settings,
        workspace_id=workspace_id,
        workspace_name=(f"{dataset.name} · {profile.target_tokens} tokens"),
    )
    database = Database(settings)
    storage = ObjectStorage(settings)
    started = time.perf_counter()
    try:
        print(
            f"[{workspace_id}] ingesting {len(dataset.selected_doc_ids)} documents",
            flush=True,
        )
        pipeline = await PipelineRunner(
            settings=settings,
            database=database,
            object_storage=storage,
        ).run(dataset.corpus_dir, workspace_id)
        if pipeline.failed_count:
            raise RuntimeError(
                f"pipeline failed for {pipeline.failed_count} files: {pipeline.errors[:3]}"
            )
        print(
            f"[{workspace_id}] indexing {len(pipeline.indexable_asset_ids)} text chunks",
            flush=True,
        )
        embedding = await AssetEmbeddingService(
            settings=settings,
            repository=EmbeddingRepository(database),
            model_client=model_client,
            vector_store=MilvusVectorStore(settings),
            artifact_reader=storage,
        ).run(
            workspace_id=workspace_id,
            embedding_type=EmbeddingType.NATIVE_MULTIMODAL,
        )
        if embedding.failed_count:
            raise RuntimeError(
                f"embedding failed for {embedding.failed_count} chunks: {embedding.errors[:3]}"
            )
    finally:
        await database.dispose()

    print(f"[{workspace_id}] running {len(dataset.queries)} queries", flush=True)
    metrics, rows = await _evaluate_workspace(
        settings=settings,
        workspace_id=workspace_id,
        queries=dataset.queries,
        query_client=query_client,
        concurrency=query_concurrency,
    )
    print(
        f"[{workspace_id}] Recall@20={metrics['recall_at_20']:.4f}, "
        f"complete evidence={metrics['complete_evidence_rate_at_20']:.4f}, "
        f"duplicate={metrics['duplicate_rate_at_20']:.4f}",
        flush=True,
    )
    report = {
        "workspace_id": workspace_id,
        "workspace_created": bootstrap.workspace_created,
        "profile": asdict(profile),
        "dataset": {
            "name": dataset.name,
            "corpus_size": len(dataset.selected_doc_ids),
            "query_count": len(dataset.queries),
            "selection_sha256": dataset.selection_sha256,
        },
        "pipeline": pipeline.model_dump(mode="json"),
        "embedding": embedding.model_dump(mode="json"),
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "queries": rows,
    }
    await asyncio.to_thread(
        _write_text,
        output_root / f"text-chunk-{profile.target_tokens}.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    return report


def _summary_markdown(reports: Sequence[Mapping[str, Any]]) -> str:
    dataset_name = reports[0]["dataset"]["name"] if reports else "unknown"
    lines = [
        "# Text chunk-size benchmark",
        "",
        f"Dataset: {dataset_name}.",
        "Retrieval channel: `native_multimodal` text embedding only. `top_k=20`.",
        "Each target uses proportionally scaled min/max/merge limits and an isolated workspace.",
        "",
        "| Target | Child chunks | Mean tokens | P95 tokens | Recall@5 | Recall@10 | "
        "Recall@20 | MRR@20 | Evidence coverage | Complete evidence | Duplicate@20 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        profile = report["profile"]
        metrics = report["metrics"]
        lines.append(
            "| {target} | {count} | {mean:.1f} | {p95} | {recall5:.4f} | "
            "{recall10:.4f} | {recall20:.4f} | {mrr20:.4f} | {coverage:.4f} | "
            "{complete:.4f} | {duplicate:.4f} |".format(
                target=profile["target_tokens"],
                count=metrics["child_chunk_count"],
                mean=metrics["mean_child_tokens"],
                p95=metrics["p95_child_tokens"],
                recall5=metrics["recall_at_5"],
                recall10=metrics["recall_at_10"],
                recall20=metrics["recall_at_20"],
                mrr20=metrics["reciprocal_rank_at_20"],
                coverage=metrics["evidence_sentence_coverage_at_20"],
                complete=metrics["complete_evidence_rate_at_20"],
                duplicate=metrics["duplicate_rate_at_20"],
            )
        )
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- Recall@K is calculated on unique source documents in the first K results, "
            "not chunk Asset IDs.",
            "- MRR@20 is the mean reciprocal rank of the first relevant source document.",
            "- Evidence coverage is the best fraction of a gold rationale's sentences "
            "present in one returned context window.",
            "- Complete evidence is the share of evidence-labeled queries whose full "
            "gold rationale appears in one returned context window.",
            "- Duplicate@20 is `1 - unique source documents / returned results` in Top 20.",
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_settings = Settings()
    if args.suite == "longbench":
        dataset = await asyncio.to_thread(
            prepare_longbench_dataset,
            longbench_root=args.longbench_root,
            output_root=args.output_root,
            per_task=args.per_task,
        )
    else:
        dataset = await asyncio.to_thread(
            prepare_dataset,
            beir_zip=args.beir_zip,
            official_root=args.official_root,
            output_root=args.output_root,
            corpus_size=args.corpus_size,
        )
    evidence_query_count = sum(bool(query.evidence) for query in dataset.queries)
    print(
        f"prepared {len(dataset.selected_doc_ids)} documents and "
        f"{len(dataset.queries)} real queries "
        f"({evidence_query_count} with evidence)",
        flush=True,
    )
    reports: list[dict[str, Any]] = []
    async with DoubaoClient(base_settings) as model_client:
        query_client = CachedQueryEmbeddingClient(model_client)
        for target in args.targets:
            cached = _load_compatible_report(
                output_root=args.output_root,
                dataset=dataset,
                target=target,
            ) if args.reuse_existing else None
            if cached is not None:
                print(f"[{cached['workspace_id']}] reusing compatible report", flush=True)
                reports.append(cached)
                continue
            reports.append(
                await _run_variant(
                    base_settings=base_settings,
                    profile=_profile(target),
                    dataset=dataset,
                    model_client=model_client,
                    query_client=query_client,
                    output_root=args.output_root,
                    query_concurrency=args.query_concurrency,
                )
            )
    summary = _summary_markdown(reports)
    await asyncio.to_thread(_write_text, args.output_root / "summary.md", summary)
    print(summary)
    return reports


def _load_compatible_report(
    *,
    output_root: Path,
    dataset: PreparedDataset,
    target: int,
) -> dict[str, Any] | None:
    path = output_root / f"text-chunk-{target}.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("dataset", {}).get("selection_sha256") != dataset.selection_sha256
        or report.get("profile", {}).get("target_tokens") != target
    ):
        return None
    rows = [_upgrade_query_row(row) for row in report["queries"]]
    ranking_metrics = aggregate_metrics(rows)
    report["queries"] = rows
    report["metrics"].update(ranking_metrics)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("scifact", "longbench"),
        default="scifact",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGETS),
    )
    parser.add_argument("--corpus-size", type=int, default=DEFAULT_CORPUS_SIZE)
    parser.add_argument("--per-task", type=int, default=60)
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=DEFAULT_QUERY_CONCURRENCY,
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse reports with the same dataset selection and target size",
    )
    parser.add_argument(
        "--beir-zip",
        type=Path,
        default=Path("data/benchmarks/beir-scifact/scifact.zip"),
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("data/benchmarks/scifact-official/data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/benchmarks/text-chunk-eval"),
    )
    parser.add_argument(
        "--longbench-root",
        type=Path,
        default=Path("data/benchmarks/longbench/data"),
    )
    args = parser.parse_args()
    if args.corpus_size < 1:
        parser.error("--corpus-size must be positive")
    if args.query_concurrency < 1:
        parser.error("--query-concurrency must be positive")
    if args.per_task < 1:
        parser.error("--per-task must be positive")
    required_paths = (
        (
            args.longbench_root / "multifieldqa_zh.jsonl",
            args.longbench_root / "dureader.jsonl",
        )
        if args.suite == "longbench"
        else (args.beir_zip, args.official_root / "corpus.jsonl")
    )
    for path in required_paths:
        if not path.exists():
            parser.error(f"required dataset path does not exist: {path}")
    return args


if __name__ == "__main__":
    asyncio.run(run(_arguments()))
