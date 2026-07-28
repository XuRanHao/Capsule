# Capsule

Capsule is a proof-of-concept pipeline for turning local Markdown, image, and
video files into multimodal assets, embeddings, and explainable HDBSCAN
clusters.

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

Search, Search Capsule, upload/delete APIs, a web frontend, multi-user
permissions, and feature editing are outside the first milestone.

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

## Concurrency defaults

| Stage | Default |
| --- | ---: |
| Asset understanding | 6 |
| Embedding generation | 16 |
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

This initial commit is the executable project skeleton. Service adapters and
the end-to-end processing implementation are added milestone by milestone
without changing the public module boundaries.
