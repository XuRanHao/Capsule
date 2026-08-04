# Capsule

Capsule is a proof-of-concept system for turning local Markdown, plain-text,
Word, PDF, image, and video files into multimodal assets, embeddings, explainable HDBSCAN
clusters, and searchable results.

The repository is split at the Asset/Embedding handoff. The role B retrieval
path is implemented end to end:

```text
text / image / image+text
  -> Query Parser
  -> concurrent query embeddings
  -> 12-channel Milvus recall
  -> PostgreSQL filters
  -> RRF or normalized-similarity fusion
  -> optional Doubao rerank
  -> deduplication and source folding
  -> Search Capsule snapshot
```

The ingestion/orchestration skeleton, Asset contracts, clustering primitives,
local runtime, full retrieval API, and web workbench live in the same project.
The non-dry-run role A import runner is still a separate implementation
boundary; retrieval starts from persisted Assets and Embedding Records.

## Core decisions

- Python 3.11+ and a single-process asynchronous runner.
- Separate concurrency pools for Asset understanding, search Query Parser,
  native embedding, text embedding, capsule naming, file parsing, and FFmpeg.
- Enrichment overlaps native embedding with Asset understanding, then runs the
  independent description and Feature embedding channels concurrently.
- PostgreSQL stores business metadata; Milvus stores vectors; S3/TOS-compatible
  object storage stores source and derived media.
- Re-imports replace a source file by `(workspace_id, relative_path)` and keep
  stable Asset IDs where the Asset locator is unchanged.
- Images referenced by Markdown preserve nearby source text separately from
  model-generated descriptions.
- Native video vectors use the rendered segment MP4 through a temporary signed
  object-storage URL; they record `embedding_source_mode=original_video`.
- Document chunk token counts run locally with the bundled DeepSeek V3 BPE
  tokenizer. Ark `/tokenization` remains available only as an explicit
  validation client and is not required for parsing documents.

## Local environment

The supported local stack is:

- Python 3.11 managed by `uv`
- Node.js 22+ and npm
- Docker Desktop
- FFmpeg and FFprobe
- PostgreSQL 17, Milvus 2.5, etcd, and MinIO from Docker Compose

The setup is idempotent. It installs backend/frontend dependencies, starts all
infrastructure, applies PostgreSQL migrations, creates the MinIO bucket and
Milvus collection, and seeds `workspace_demo`.

```bash
make setup
```

Then put the real Volcengine Ark key in the ignored local `.env` file:

```dotenv
CAPSULE_ARK_API_KEY=your-ark-api-key
# 可选：覆盖项目内置的 DeepSeek V3 tokenizer.json
# CAPSULE_DOCUMENT_TOKENIZER_PATH=/absolute/path/to/tokenizer.json
# 子块以 400 token 为目标，低于 250 时与邻块合并，500 是结构化软上限
CAPSULE_DOCUMENT_CHUNK_MIN_TOKENS=250
CAPSULE_DOCUMENT_CHUNK_TARGET_TOKENS=400
CAPSULE_DOCUMENT_CHUNK_MAX_TOKENS=500
CAPSULE_DOCUMENT_CHUNK_MERGE_MAX_TOKENS=600
CAPSULE_DOCUMENT_PARENT_MAX_TOKENS=2000
# Word/PDF 内嵌图片在本地抽取，并按需使用 RapidOCR
CAPSULE_DOCUMENT_OCR_ENABLED=true
```

Start the API and frontend together:

```bash
make dev
```

Open `http://localhost:3000`. The API is at `http://localhost:8010`, its
interactive documentation is at `http://localhost:8010/docs`, MinIO is at
`http://localhost:9001`, and Milvus listens on `localhost:19530`.

Useful commands:

```bash
make status       # configuration, containers, and API health
make bootstrap    # re-run migrations and idempotent resource initialization
make test         # backend and frontend quality gates
make down         # stop containers without deleting persisted data
```

The assetization CLI persists Markdown, plain-text, Word, PDF, image and video Assets to
PostgreSQL. Video segments additionally persist playable MP4/JPEG artifacts in
MinIO/S3. A failed file is recorded without aborting the batch. Use the `embed`
CLI command after import to write `EmbeddingRecord` metadata and vectors to
Milvus.

Repeated imports reuse a completed source when its workspace-relative path,
SHA-256 digest, and assetization fingerprint are unchanged. The fingerprint
includes parser configuration that affects generated Assets. Increment
`CAPSULE_ASSETIZATION_VERSION` after changing parser behavior or media output
semantics so unchanged source bytes are processed once with the new logic.

To reset persisted PostgreSQL/Milvus/MinIO data, use
`docker compose down -v` deliberately; `make down` preserves it.

## Manual quick start

1. Copy the environment template.

   ```bash
   cp .env.example .env
   ```

   Model workloads are configured independently: `CAPSULE_SEARCH_PARSER_MODEL`
   controls Query Parser calls, `CAPSULE_UNDERSTANDING_MODEL` controls asset and
   cluster understanding, and `CAPSULE_EMBEDDING_MODEL` controls the shared
   multimodal embedding space. Changing the embedding model requires rebuilding
   its indexes; changing only the Parser model does not.

2. Start local infrastructure.

   ```bash
   docker compose up -d
   ```

3. Create a virtual environment and install the project.

   ```bash
   uv sync --extra dev
   . .venv/bin/activate
   ```

4. Apply migrations and initialize all storage resources.

   ```bash
   uv run capsule bootstrap --workspace workspace_demo
   ```

5. Inspect the configuration and scan a test directory.

   ```bash
   capsule doctor
   capsule scan ./test-data
   capsule pipeline ./test-data --workspace workspace_demo --dry-run
   capsule pipeline ./test-data --workspace workspace_demo --execute
   capsule embed --workspace workspace_demo
   ```

The non-dry-run pipeline persists Markdown, plain-text, Word, PDF, image and video Assets
to PostgreSQL. Video input requires the host MPS worker described below; a
video failure is recorded without aborting the rest of the batch.

Browser imports stream each committed Asset into a bounded enrichment queue
instead of waiting for the complete folder. Native embedding starts alongside
Understanding; that Asset's text embedding channels start only after its
description and Feature fields have been committed. The import Job is finalized
only after both file processing and the enrichment queue have drained.

`capsule embed` processes one embedding route at a time. Its default is native
multimodal: Markdown uses the block text, images are sent inline as their
original bytes, and video uses the derived playable MP4 inline as a Base64 Data
URI. Repeat `--asset-id` to limit a batch; already indexed logical inputs are
skipped unless `--force` is set.

## Video MPS worker

Video visual features run only on the macOS host because Docker cannot access
MPS. A single FFmpeg decode uses VideoToolbox on macOS and emits only 6 fps of
224px analysis data instead of transferring every full-resolution frame into
Python. It measures activity at 6 fps and samples a centered 224x224 frame every
0.5 seconds for JPEG caching and MobileCLIP-S0 embeddings. Time-constrained
content clustering derives a
per-video first-stage distance threshold from the adjacent-distance Q75. A
second stage greedily merges content-compatible neighbors using adaptive
duration and sustained activity-shift costs. The cached embeddings select up
to three quality-filtered representative frames, so keyframes require no
second extraction or MobileCLIP pass. Each final Segment is then rendered once
as a playable MP4 while its selected cached 224x224 JPEGs become the cover and
representative keyframes. The derived media is written to the configured private
S3-compatible bucket; the Asset keeps the logical time range plus generic
`derived_file_uri`, `preview_uri`, and video-specific `file_info.keyframes`.
Both downstream video model paths read their durable private `s3://` objects
from MinIO and send them inline to Ark as Base64 Data URIs: Understanding
receives up to three representative JPEG keyframes, while native multimodal
Embedding receives the rendered Segment MP4. Ark therefore does not need
network access to the MinIO endpoint.

Run the command from a native Apple-silicon Python environment that has the
Capsule dependencies plus PyTorch, Apple `ml-mobileclip`, FFmpeg and FFprobe:

```bash
uv run capsule mps-video data/dev-fixtures/nature/hiking-trip.mp4 \
  --workspace workspace_demo
```

The MobileCLIP-S0 checkpoint defaults to
`data/models/mobileclip-s0/mobileclip_s0.pt` and can be changed with
`CAPSULE_MOBILECLIP_MODEL_PATH`. The command fails clearly when MPS, FFmpeg,
FFprobe, MobileCLIP or the checkpoint is unavailable; it never silently uses
Docker CPU.

A long-running `PipelineRunner` owns one lazy, process-resident MobileCLIP
worker. The model is loaded by the first video and reused by later import jobs;
concurrent video analysis cannot initialize duplicate MPS model copies. Runs
that contain only images or documents do not load the model. A one-shot CLI
process still releases the model when that process exits.

Video rendering and object-storage upload use separate bounded pools. FFmpeg
writes one Segment bundle (MP4, preview source, and representative keyframes)
to `CAPSULE_VIDEO_SPOOL_ROOT`, publishes its manifest to Redis Streams, and
releases the FFmpeg slot immediately. Upload workers retry deterministic object
keys and reclaim abandoned pending messages with `XAUTOCLAIM`. A source-level
generation guard makes repeated delivery idempotent and rejects stale work.
Each successfully uploaded Segment is committed and submitted to Understanding
immediately; generation finalization removes obsolete Segments only after the
whole source succeeds. Set `CAPSULE_VIDEO_UPLOAD_QUEUE_BACKEND=memory` to run
the bounded in-process transport for local comparison tests.

Without Homebrew, this checkout can use the ignored project-local binaries at
`tmp/tools/ffmpeg/bin/`; the macOS parser discovers them automatically. Docker
continues to use its Linux FFmpeg instead.

6. Start the search API after configuring `CAPSULE_ARK_API_KEY`.

   ```bash
   uv run uvicorn capsule.api.app:app --host 0.0.0.0 --port 8010
   ```

7. Start the role B search workspace in another terminal.

   ```bash
   cd frontend
   cp .env.example .env.local
   npm ci
   npm run dev
   ```

   Open `http://localhost:3000`. The page supports text, uploaded image,
   image URL, and combined image-text queries. It exposes precise/quick mode,
   both fusion algorithms, optional reranking, all documented filters, Query
   Parser output, channel evidence, source folding, and Search Capsules.

   Search accepts text, image, and combined image-text queries:

   ```bash
   curl http://localhost:8010/api/v1/search \
     -H 'Content-Type: application/json' \
     -d '{
       "workspace_id": "workspace_demo",
       "query_type": "text",
       "query_text": "蓝紫色黄昏动画场景",
       "precision_mode": true,
       "fusion_method": "weighted_rrf",
       "rerank": "doubao_seed_2_lite",
       "save_capsule": true,
       "filters": {"asset_type": ["image", "video_segment"]},
       "top_k": 20
     }'
   ```

   For uploaded image queries, call `POST /api/v1/query-images` first and pass
   the returned `upload_id` as `query_image_upload_id`. Doubao must be able to
   fetch the generated URL, so local MinIO uploads require
   `CAPSULE_OBJECT_STORAGE_PUBLIC_ENDPOINT`; a public image URL needs no tunnel.

   Search Capsule APIs:

   ```text
   GET    /api/v1/search-capsules
   GET    /api/v1/search-capsules/{id}
   POST   /api/v1/search-capsules/{id}/refresh
   PATCH  /api/v1/search-capsules/{id}
   DELETE /api/v1/search-capsules/{id}
   ```

   Run the documented relevance gates with a labeled JSONL file:

   ```bash
   uv run capsule evaluate-search evaluation.jsonl --strict
   ```

   Each line contains `{"request": {...}, "relevant_asset_ids": ["asset_..."]}`.
   The command reports Precision@5 and Recall@10 overall and by query type.

## Search pipeline

```text
query
  -> multimodal Query Parser and explicit weighted dimensions
  -> bounded concurrent query embeddings
  -> concurrent Milvus recall scoped by workspace and Asset metadata
  -> PostgreSQL-authoritative hydration, indexed-status and revision validation
  -> PostgreSQL favorite / cluster filters and exact Asset field recheck
  -> weighted RRF or normalized weighted similarity
  -> optional Seed-2-lite rerank of the top 30
  -> exact dedup, same-source cap and video / Markdown folding
  -> Search Capsule execution and immutable result snapshot
```

The available embedding routes are `native_multimodal`, `asset_description`,
and all ten Asset Feature dimensions. Requests default to `native_multimodal`
only and may explicitly select multiple routes with `embedding_types`; selected
routes receive equal weights in quick mode. A native-only request always skips
model parsing; multi-route precision parsing rewrites only the selected routes
with model thinking disabled. For text and image-text precision queries, explicit
dimension preferences in the query determine normalized route weights; pure-image
precision queries remain equal-weighted. Image-text search preserves image constraints and applies text additions,
modifications, and exclusions through late fusion and reranking. Any failed route
is removed, surviving weights are renormalized, and the response reports
`degraded=true`.

Search dimensions are validated against the target `filters.asset_type` values.
`visual_style` and `color_composition` are available only for images and video
segments; Markdown and plain-text assets skip those channels during indexing.
Mixed target-type searches use union semantics, so visual channels remain
available but apply only to their image/video subset.

Search filter names follow PostgreSQL columns. In particular, use
`filters.model_name` for the Embedding model and `filters.file_type` for the
Asset extension. The former `embedding_model_version` request key remains
accepted only as an input compatibility alias.

## Concurrency defaults

| Stage | Default |
| --- | ---: |
| Asset understanding | 32 |
| Pending asset enrichment queue | 64 |
| Search query understanding | 4 |
| Native embedding generation | 24 |
| Text embedding generation | 96 |
| Search query embedding | 16 |
| Cluster naming | 8 |
| File parsing | 4 |
| FFmpeg | 2 |

All values are environment-driven. HTTP 429, transient 5xx errors, and network
timeouts are retried with exponential backoff and jitter.

The 2026-07-29 local Ark benchmark measured Asset Understanding throughput at
20.80, 29.10, 36.03, 40.32, and 45.64 Assets/minute for concurrency 16, 24,
32, 36, and 40 respectively. The recommended shared-host configuration is
Asset Understanding 32, Search Understanding 4, Native Embedding 16, and Text
Embedding 16. Concurrency 40 remains a dedicated-worker load-test ceiling
until full video keyframe, object-storage, and PostgreSQL pressure is
validated. On nine real images, the earlier bounded Understanding-to-Embedding
pipeline reduced full enrichment time from 45.60 seconds to 43.28 seconds
(5.1%) with no recorded failures; the current upstream pipeline additionally
overlaps native embedding and fans out all text channels concurrently.
The committed Asset-stored streaming pipeline was then measured in two
cross-ordered live Ark runs over three images and three Markdown files. Average
end-to-end time fell from 13.008 seconds to 10.911 seconds (16.12%); both runs
produced all six descriptions and Feature objects without processing errors.

## Development

```bash
ruff check .
pytest
mypy src/capsule
```

## Repository status

The role B retrieval and Search Capsule chain is executable. Real relevance
targets must be measured with the project evaluation set and a valid Ark key;
unit and integration tests verify behavior, not Precision@5/Recall@10 quality.
