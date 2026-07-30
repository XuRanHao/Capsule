# Capsule

Capsule is a proof-of-concept system for turning local Markdown, plain-text,
image, and video files into multimodal assets, embeddings, explainable HDBSCAN
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
# Markdown 与 TXT 均使用这个上限，默认值也是 400
CAPSULE_DOCUMENT_CHUNK_MAX_TOKENS=400
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

The assetization CLI persists Markdown, plain-text, image and video Assets to
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

The non-dry-run pipeline persists Markdown, plain-text, image and video Assets
to PostgreSQL. Video input requires the host MPS worker described below; a
video failure is recorded without aborting the rest of the batch.

`capsule embed` processes one embedding route at a time. Its default is native
multimodal: Markdown uses the block text, images are sent inline as their
original bytes, and video uses the derived playable MP4 through a temporary
signed object-storage URL. Repeat `--asset-id` to limit a batch; already
indexed logical inputs are skipped unless `--force` is set. For Ark to process
videos, `CAPSULE_OBJECT_STORAGE_PUBLIC_ENDPOINT` must point to an endpoint Ark
can reach; the local Docker-only MinIO address cannot be used for this step.

## Video MPS worker

Video visual features run only on the macOS host because Docker cannot access
MPS. The first pass uses scene detection, 45-second long-shot splitting into
20-second windows, end-inclusive 5-second frame sampling (maximum 12), quality
filtering, and up to three MobileCLIP-S0 representative frames. Each final
Segment is then rendered once as a playable MP4, cover image and representative
keyframe images. The derived media is written to the configured private
S3-compatible bucket; the Asset keeps the logical time range plus generic
`derived_file_uri`, `preview_uri`, and video-specific `file_info.keyframes`.

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
and all ten Asset Feature dimensions. Text fallback uses the documented six
routes. Quick image search uses native multimodal only; precise image search
parses the image into six weighted routes. Image-text search preserves image
constraints and applies text additions, modifications, and exclusions through
late fusion and reranking. Any failed route is removed, surviving weights are
renormalized, and the response reports `degraded=true`.

Search filter names follow PostgreSQL columns. In particular, use
`filters.model_name` for the Embedding model and `filters.file_type` for the
Asset extension. The former `embedding_model_version` request key remains
accepted only as an input compatibility alias.

## Concurrency defaults

| Stage | Default |
| --- | ---: |
| Asset understanding | 32 |
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
