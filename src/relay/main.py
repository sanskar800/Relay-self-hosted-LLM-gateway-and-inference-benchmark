"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to a backend."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from relay.api import chat
from relay.api.errors import install_error_handlers
from relay.config import Settings


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the app. `transport` lets tests replace the network with a fake backend."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client (and connection pool) for the whole process, closed on shutdown.
        # For streams, the read timeout applies to each read, i.e. it is the maximum
        # silence between chunks, not the total duration.
        async with httpx.AsyncClient(
            base_url=str(settings.backend_url),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
            transport=transport,
        ) as client:
            app.state.backend = client
            yield

    app = FastAPI(title="Relay", lifespan=lifespan)
    install_error_handlers(app)
    app.include_router(chat.router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about backends (that is /readyz, Day 4)."""
        return {"status": "ok"}

    return app


app = create_app()
