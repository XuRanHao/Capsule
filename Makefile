.PHONY: setup infra bootstrap dev status test down

setup:
	./scripts/setup-local.sh

infra:
	docker compose up -d --wait

bootstrap:
	uv run capsule bootstrap --workspace workspace_demo --workspace-name "Capsule Demo"

dev:
	./scripts/dev-local.sh

status:
	./scripts/check-local.sh

test:
	uv run ruff check .
	uv run mypy src/capsule
	uv run pytest
	cd frontend && npm run lint && npm test

down:
	docker compose down
