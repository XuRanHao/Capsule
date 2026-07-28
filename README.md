# Capsule

Capsule is a proof-of-concept backend for turning local Markdown, image, and
video files into multimodal assets, embeddings, explainable HDBSCAN clusters,
and searchable results.

The first milestone deliberately focuses on one complete backend path:

```text
local files
  -> source files
  -> semantic assets
  -> description and features
  -> embeddings
  -> Milvus
  -> PCA + HDBSCAN
  -> cluster capsules
```

Search Capsule, upload/delete APIs, a web frontend, multi-user permissions,
and feature editing are outside the current milestone. The search backend is
developed independently against the frozen PostgreSQL Asset and Milvus
embedding contracts.

## Core decisions

- Python 3.11+ and a single-process asynchronous runner.
- Separate concurrency limits for understanding, embedding, capsule naming,
  file parsing, and FFmpeg.
- PostgreSQL stores business metadata; Milvus stores vectors; S3/TOS-compatible
  object storage stores source and derived media.
- Duplicate files are skipped by `(workspace_id, sha256)`.
- Images referenced by Markdown preserve nearby source text separately from
  model-generated descriptions.
- Video segments initially use description-based embeddings and record
  `embedding_source_mode=description_fallback`.

## Quick start

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

4. Inspect the configuration and scan a test directory.

   ```bash
   capsule doctor
   capsule scan ./test-data
   capsule pipeline ./test-data --workspace workspace_demo --dry-run
   ```

The non-dry-run pipeline command is intentionally guarded until concrete
storage, model, and vector-store adapters are wired in.

5. Start the search API after configuring `CAPSULE_ARK_API_KEY`.

   ```bash
   uv run uvicorn capsule.api.app:app --host 0.0.0.0 --port 8000
   ```

6. Start the role B search workspace in another terminal.

   ```bash
   cd frontend
   cp .env.example .env.local
   npm install
   npm run dev
   ```

   Open `http://localhost:3000`. The page supports text, image URL, and
   combined image-text queries and renders channel scores, source files, video
   timecodes, and the `source_contexts` paragraph associated with every asset.

   Search accepts text, image, and combined image-text queries:

   ```bash
   curl http://localhost:8000/api/v1/search \
     -H 'Content-Type: application/json' \
     -d '{
       "workspace_id": "workspace_demo",
       "query_type": "text",
       "query_text": "蓝紫色黄昏动画场景",
       "filters": {"asset_type": ["image", "video_segment"]},
       "top_k": 20
     }'
   ```

## Search pipeline

```text
query
  -> bounded concurrent query embeddings
  -> concurrent Milvus channel recall
  -> weighted reciprocal-rank fusion
  -> exact-asset deduplication and same-source cap
  -> one PostgreSQL batch hydration
  -> explainable API response
```

Text queries recall `native_multimodal`, `asset_description`,
`subject_content`, and `visual_style` concurrently. Image queries use the
native multimodal channel. Image-text queries prefer a joint embedding and
fall back to separately embedded image and text vectors. A failed recall
channel does not fail the whole request and is reported with
`degraded=true`.

## Concurrency defaults

| Stage | Default |
| --- | ---: |
| Asset understanding | 6 |
| Embedding generation | 16 |
| Search query embedding | 16 |
| Cluster naming | 4 |
| File parsing | 4 |
| FFmpeg | 2 |

All values are environment-driven. HTTP 429, transient 5xx errors, and network
timeouts are retried with exponential backoff and jitter.

## Development

```bash
ruff check .
pytest
mypy src/capsule
```

## Repository status

The repository contains the executable project skeleton and the role B search
backend. The ingestion implementation is added milestone by milestone without
changing the frozen handoff contracts.
