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

CAPSULE_API_EMBEDDED_CPU_TASKS_ENABLED=false uv run uvicorn capsule.api.app:app \
  --host 0.0.0.0 \
  --port 8010 \
  --reload &
backend_pid=$!

npm --prefix frontend run dev &
frontend_pid=$!

# Browser imports commit durable PostgreSQL tasks. Run every matching recovery
# scheduler and worker locally so a task cannot remain queued after `complete`.
uv run capsule video-scheduler &
video_scheduler_pid=$!
uv run capsule video-worker --worker-id dev-video-worker &
video_worker_pid=$!
uv run capsule cpu-task-scheduler --kind image &
image_scheduler_pid=$!
uv run capsule cpu-task-worker --kind image --worker-id dev-image-worker &
image_worker_pid=$!
uv run capsule cpu-task-scheduler --kind text &
text_scheduler_pid=$!
uv run capsule cpu-task-worker --kind text --worker-id dev-text-worker &
text_worker_pid=$!

pids=(
  "$backend_pid" "$frontend_pid"
  "$video_scheduler_pid" "$video_worker_pid"
  "$image_scheduler_pid" "$image_worker_pid"
  "$text_scheduler_pid" "$text_worker_pid"
)

cleanup() {
  kill "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Capsule API: http://localhost:8010"
echo "Capsule Web: http://localhost:3000"
echo "Durable workers: video, image, text (with recovery schedulers)"
wait -n "${pids[@]}"
