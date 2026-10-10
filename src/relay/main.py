"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to backends."""

import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from relay.accounting.usage import UsageRecord
from relay.api import chat, models
from relay.api.errors import install_error_handlers
from relay.backends import ENGINES, Backend
from relay.config import RelayConfig, Settings, load_config

log = logging.getLogger("relay.usage")


def configure_logging() -> None:
    """Minimal console logging for Relay's own loggers (structured JSON logs come later).

    uvicorn configures only its own loggers, so without this our INFO lines are dropped.
    """
    logger = logging.getLogger("relay")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def log_usage(record: UsageRecord) -> None:
    """Default usage sink until the usage ledger exists: one log line per request."""
    log.info("usage %s", record)


def build_backends(
    config: RelayConfig, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Backend]:
    """One Backend object (with its own connection pool) per configured backend."""
    return {
        name: ENGINES[cfg.engine](
            name,
            str(cfg.url),
            connect_timeout_s=cfg.connect_timeout_s,
            read_timeout_s=cfg.read_timeout_s,
            transport=transport,
        )
        for name, cfg in config.backends.items()
    }


def create_app(
    settings: Settings | None = None,
    *,
    config: RelayConfig | None = None,
    backends: Mapping[str, Backend] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    usage_sink: Callable[[UsageRecord], None] = log_usage,
) -> FastAPI:
    """Build the app.

    The config is loaded and validated here, at startup, so a bad file stops the process
    before it serves anything. Tests can pass a `config`, ready-made fake `backends`
    (no HTTP at all), or a `transport` that replaces the network under real backends.
    """
    configure_logging()
    config = config or load_config((settings or Settings()).config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.backends = dict(backends) if backends else build_backends(config, transport)
        missing = set(config.backends) - set(app.state.backends)
        if missing:
            raise RuntimeError(f"no backend object for configured backends: {sorted(missing)}")
        app.state.started_at = int(time.time())
        # Called once per completed request with its token usage.
        app.state.usage_sink = usage_sink
        try:
            yield
        finally:
            for backend in app.state.backends.values():
                await backend.aclose()

    app = FastAPI(title="Relay", lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(chat.router)
    app.include_router(models.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about backends (that is /readyz, Day 4)."""
        return {"status": "ok"}

    return app


app = create_app()
