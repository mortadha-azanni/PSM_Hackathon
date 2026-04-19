FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy source code
COPY . .

# Install the project
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
