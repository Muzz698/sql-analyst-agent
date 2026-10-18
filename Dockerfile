# syntax=docker/dockerfile:1
# One image, two entry points:
#   default            -> Streamlit UI (in-process agent; what Hugging Face Spaces runs)
#   docker-compose api -> FastAPI backend (UI then talks to it via API_URL)
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# HF Spaces runs containers as uid 1000; use the same user everywhere.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=7860

# 1) Dependencies only (cached until pyproject.toml / uv.lock change).
COPY pyproject.toml uv.lock README.md ./
# Cache mount keeps uv's download cache out of the image (~480 MB) and speeds rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2) Application code + data.
COPY src ./src
COPY data/download_chinook.py ./data/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev \
    && python data/download_chinook.py \
    && mkdir -p /app/state \
    && chown -R app:app /app/data /app/state

USER app
EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"

CMD ["sh", "-c", "streamlit run src/sql_analyst/ui/app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]
