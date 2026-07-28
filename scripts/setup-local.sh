#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

for command_name in docker uv node npm ffmpeg ffprobe; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "缺少运行依赖: $command_name" >&2
    exit 1
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "Docker 尚未启动，请先打开 Docker Desktop。" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi
if [[ ! -f frontend/.env.local ]]; then
  cp frontend/.env.example frontend/.env.local
fi

uv sync --extra dev
SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm --prefix frontend ci --no-audit --no-fund
docker compose up -d --wait
uv run capsule bootstrap --workspace workspace_demo --workspace-name "Capsule Demo"

if [[ -z "${CAPSULE_ARK_API_KEY:-}" ]] \
  && ! grep -Eq '^CAPSULE_ARK_API_KEY=.+$' .env; then
  echo
  echo "基础环境已就绪。请在 .env 填写 CAPSULE_ARK_API_KEY 后运行 make dev。"
else
  echo
  echo "环境与模型凭据均已就绪，运行 make dev 启动完整工作台。"
fi
