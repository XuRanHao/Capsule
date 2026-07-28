#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -f .env || ! -f frontend/.env.local ]]; then
  echo "本地环境文件缺失，请先运行 make setup。" >&2
  exit 1
fi
if [[ -z "${CAPSULE_ARK_API_KEY:-}" ]] \
  && ! grep -Eq '^CAPSULE_ARK_API_KEY=.+$' .env; then
  echo "请先在 .env 填写 CAPSULE_ARK_API_KEY。" >&2
  exit 2
fi

docker compose up -d --wait
uv run capsule bootstrap --workspace workspace_demo --workspace-name "Capsule Demo"

uv run uvicorn capsule.api.app:app \
  --host 0.0.0.0 \
  --port 8010 \
  --reload &
backend_pid=$!

npm --prefix frontend run dev &
frontend_pid=$!

cleanup() {
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Capsule API: http://localhost:8010"
echo "Capsule Web: http://localhost:3000"
wait
