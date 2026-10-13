# Relay gateway image. Built by Compose (`make up-all`) or: docker build -t relay .
# Exact dependency versions come from uv.lock; runs `python -m relay.serve`, so
# `docker stop` (SIGTERM) triggers the graceful drain.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
COPY config ./config

RUN useradd --create-home --uid 10001 relay
USER relay

ENV PATH="/app/.venv/bin:$PATH" \
    RELAY_HOST=0.0.0.0 \
    RELAY_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --retries=20 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["python", "-m", "relay.serve"]
