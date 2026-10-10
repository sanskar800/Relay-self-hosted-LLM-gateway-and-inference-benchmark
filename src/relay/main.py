"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to a backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from relay.api import chat
from relay.api.errors import install_error_handlers
from relay.backends import Backend, LlamaCppBackend
from relay.config import Settings


def create_app(
    settings: Settings | None = None,
    *,
    backend: Backend | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the app.

    Tests can inject a fake `backend` (no HTTP at all), or a `transport` that replaces
    the network under the real HTTP backend.
    """
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One backend (and connection pool) for the whole process, closed on shutdown.
        app.state.backend = backend or LlamaCppBackend(
            "llamacpp",
            str(settings.backend_url),
            connect_timeout_s=settings.connect_timeout_s,
            read_timeout_s=settings.request_timeout_s,
            transport=transport,
        )
        try:
            yield
        finally:
            await app.state.backend.aclose()

    app = FastAPI(title="Relay", lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(chat.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about backends (that is /readyz, Day 4)."""
        return {"status": "ok"}

    return app


app = create_app()
