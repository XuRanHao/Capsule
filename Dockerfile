FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/capsule-venv \
    PATH="/opt/capsule-venv/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra dev --no-install-project

COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN uv sync --frozen --extra dev

CMD ["sleep", "infinity"]
