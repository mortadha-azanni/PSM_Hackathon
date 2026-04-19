FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
