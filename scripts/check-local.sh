#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

echo "基础配置"
uv run capsule doctor

echo
echo "基础服务"
docker compose ps

echo
if curl --fail --silent http://localhost:8010/health >/dev/null; then
  echo "搜索 API: ready"
else
  echo "搜索 API: 未启动（运行 make dev）"
fi
