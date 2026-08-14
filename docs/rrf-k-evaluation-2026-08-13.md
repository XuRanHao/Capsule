# Weighted RRF K evaluation and decision

Date: 2026-08-13
Decision: use `RRF_K=20` as the Capsule search default.

## Scope

Capsule uses weighted reciprocal rank fusion:

```text
score(asset) = sum(channel_weight / (K + channel_rank))
```

The selected search dimensions retain their normalized query weights. Local
text recall is merged as one additional recall route. This decision changes
only `K`; it does not change dimension-weight generation, the per-channel
recall limit, or the fusion candidate cap.

## Evaluation evidence

Two evaluations were run.

### Existing Capsule image workspace

- Corpus: 50 enriched and indexed images.
- Queries: 16 manually labelled Chinese multi-dimensional queries.
- Dimensions: native multimodal, subject content, scene theme, and visual
  presentation.
- The enhanced dimension queries and weights were generated once per query;
  every K value reused the same recalls.
- K values: 1, 5, 10, 20, 30, 60, 90, and 120.

The small workspace favored K=5/10 for head ranking. K=20 still materially
outperformed K=60 on NDCG@10 while preserving Recall@10.

### Nested COCO scale evaluation

- Corpus sizes: 50, 100, 250, 500, and 1000 images.
- Queries: 40 fixed queries selected from the smallest nested corpus.
- Adding data only introduced real-image distractors; query identities did not
  change with corpus size.
- Local MobileCLIP-S0 embeddings were used to avoid API/model randomness.
- Weighted vector channels: native 0.4, subject 0.3, scene 0.3.
- Dimension vectors: 0.3 native image + 0.7 dimension text.
- Production-relevant fusion mode: weighted vector RRF plus local text.
- Per-channel recall limit: 100; fusion candidate cap: 300.

Cross-scale averages for the main candidates:

| K | R@1 | R@5 | R@10 | R@20 | MRR@100 | NDCG@5 | NDCG@10 | NDCG@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 76.5% | 94.5% | 97.0% | 98.0% | 0.8442 | 0.8659 | 0.8744 | 0.8770 |
| 20 | 76.5% | 95.0% | 97.0% | 98.0% | 0.8443 | 0.8677 | 0.8744 | 0.8772 |
| 60 | 73.5% | 95.5% | 97.0% | 98.5% | 0.8274 | 0.8569 | 0.8620 | 0.8659 |

K=20 was the most balanced setting in the scale evaluation: it tied the best
NDCG@10, led NDCG@5, NDCG@20, and MRR@100, and retained the best Recall@10
band. K=60 gave a small deep-recall gain but degraded first-page ordering and
MRR. Total corpus size did not imply a monotonic best K; recall depth and
cross-channel rank disagreement were more direct drivers.

## Decision rationale

Use K=20 because it balances first-page ordering and recall across both the
existing Capsule data and the larger controlled scale evaluation. Do not
dynamically change K solely from total asset count. Re-evaluate the value when
any of these conditions materially change:

- per-channel recall depth (currently capped at 100),
- local text route weighting,
- dimension-weight generation,
- embedding model or fused-vector construction,
- labelled production query distribution.

## Reproduction artifacts

- Existing-workspace cases:
  `data/benchmarks/rrf-k-eval-50-2026-08-13.json`
- Existing-workspace result:
  `data/benchmarks/rrf-k-eval-results-2026-08-13.json`
- Scale result with extended metrics:
  `data/benchmarks/rrf-k-scale-coco-1000-2026-08-13.json`
- Existing-workspace runner: `scripts/benchmark_rrf_k.py`
- Scale runner: `scripts/benchmark_rrf_k_scale.py`

## Native/original-content proportion evaluation

The native-content proportion is not an RRF route weight. For each non-native
dimension, Capsule first builds a normalized fused vector on both sides of the
search:

```text
dimension_search_vector = normalize(
    native_content_weight * normalize(native_vector)
    + (1 - native_content_weight) * normalize(dimension_vector)
)
```

The current default is `native_content_weight=0.3`. To isolate this parameter,
the evaluation fixed `RRF_K=20`, route weights, recall depth, and candidate cap,
then swept the native-content proportion from 0.0 through 1.0 in increments of
0.1. Corpus and query vectors were both rebuilt for every value.

### Existing Capsule workspace result

On the 50-image, 16-query labelled workspace, lower native proportions gave
better first-page ordering. The important points were:

| Native proportion | Hit@1 | Recall@5 | Recall@10 | NDCG@10 | NDCG@20 |
|---:|---:|---:|---:|---:|---:|
| 0% | 100.0% | 95.18% | 98.21% | 0.9732 | 0.9775 |
| 10% | 93.75% | 96.43% | 98.21% | 0.9597 | 0.9640 |
| 30% (current) | 93.75% | 96.43% | 98.21% | 0.9550 | 0.9591 |
| 50% | 93.75% | 96.43% | 98.21% | 0.9554 | 0.9595 |
| 70% | 93.75% | 96.43% | 98.21% | 0.9478 | 0.9519 |
| 100% | 93.75% | 93.30% | 97.32% | 0.9276 | 0.9361 |

Pure dimension vectors (0%) led NDCG and Hit@1, while 10%-50% formed the best
Recall@5 plateau. Because this dataset is small and local text is strong, 0%
should be treated as a useful boundary result, not a safe new default.

### Nested 50-1000 image scale result

The COCO evaluation reused the same 1000 images and fixed 40 queries from the
RRF scale test. Nine nested corpus sizes were evaluated: 50, 100, 150, 200,
250, 350, 500, 750, and 1000. Raw image, subject-text, and scene-text vectors
were retained, so both corpus and query fusion could be recomputed. No
additional images were added. The production-relevant vector-plus-text mode
produced these best NDCG@10 points:

| Corpus size | Best native proportion | Best NDCG@10 | NDCG@10 at 30% | Recall@5 at 30% | Candidate recall at 30% |
|---:|---:|---:|---:|---:|---:|
| 50 | 50%-60% | 0.9730 | 0.9644 | 97.5% | 100.0% |
| 100 | 60%-80% | 0.9565 | 0.9381 | 97.5% | 100.0% |
| 150 | 60% | 0.9473 | 0.9196 | 97.5% | 100.0% |
| 200 | 50% | 0.9256 | 0.8979 | 97.5% | 100.0% |
| 250 | 50% | 0.9081 | 0.8896 | 97.5% | 100.0% |
| 350 | 40% | 0.8781 | 0.8716 | 95.0% | 100.0% |
| 500 | 60%-70% | 0.8394 | 0.8312 | 95.0% | 100.0% |
| 750 | 100% | 0.8118 | 0.7663 | 95.0% | 100.0% |
| 1000 | 100% | 0.7907 | 0.7573 | 87.5% | 100.0% |

Across the nine sizes, 60% had the best average vector-plus-text NDCG@10
(0.8845), compared with 0.8707 at 30%. However, only 30%-40% maintained 100%
candidate recall at every tested size. The best head-ranking proportion was
not monotonic with corpus size: it moved from 60% at 150 images to 50% at 200,
40% at 350, 60%-70% at 500, and 100% at 750-1000. The apparent preference for
high native weight at the largest sizes is also partly specific to the
controlled task: each query is a caption for one exact target image, which
directly rewards the native image-text route more than broad multi-relevant
searches do.

### Native-proportion decision

Retain `native_content_weight=0.3` for now. The evidence supports a robust
30%-50% band, but not a single replacement value across the two datasets:

- the labelled Capsule queries favor lower native proportions for NDCG;
- the larger exact-image task favors 50%-70% on average for head ranking;
- 30%-40% is strongest for candidate-recall stability across scale;
- total corpus size alone is not sufficient grounds for dynamically changing
  the proportion.

A production-weight change should wait for a larger labelled Capsule query set.
The next useful comparison is 0.3 versus 0.5 with at least 100-200 real queries,
split by exact-item, semantic-category, scene/theme, and visual-style intent.

Additional artifacts:

- Existing-workspace result:
  `data/benchmarks/native-content-weight-real-50-2026-08-13.json`
- Scale result:
  `data/benchmarks/native-content-weight-scale-coco-1000-2026-08-13.json`
- Existing-workspace runner: `scripts/benchmark_native_content_weight.py`
- Scale runner: `scripts/benchmark_native_content_weight_scale.py`
