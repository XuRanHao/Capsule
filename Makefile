.PHONY: setup infra bootstrap dev api worker-video scheduler-video worker-image \
	worker-text scheduler-image scheduler-text status test down

setup:
	./scripts/setup-local.sh

infra:
	docker compose up -d --wait

bootstrap:
	uv run capsule bootstrap --workspace workspace_demo --workspace-name "Capsule Demo"

dev:
	./scripts/dev-local.sh

# Production processes: run the API and every matching scheduler/worker as
# independent supervised services. `make dev` starts this complete set locally.
api:
	CAPSULE_API_EMBEDDED_CPU_TASKS_ENABLED=false uv run uvicorn capsule.api.app:app --host 0.0.0.0 --port 8010

worker-video:
	uv run capsule video-worker --worker-id video-worker-1

scheduler-video:
	uv run capsule video-scheduler

worker-image:
	uv run capsule cpu-task-worker --kind image --worker-id image-worker-1

worker-text:
	uv run capsule cpu-task-worker --kind text --worker-id text-worker-1

scheduler-image:
	uv run capsule cpu-task-scheduler --kind image

scheduler-text:
	uv run capsule cpu-task-scheduler --kind text

status:
	./scripts/check-local.sh

test:
	uv run ruff check .
	uv run mypy src/capsule
	uv run pytest
	cd frontend && npm run lint && npm test

down:
	docker compose down
