"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to a backend."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from relay.config import Settings


def openai_error(status_code: int, message: str, error_type: str) -> JSONResponse:
    """Error body in the shape OpenAI clients parse: {"error": {...}}."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": None, "code": None}},
    )


def create_app(
    settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    """Build the app. `transport` lets tests replace the network with a fake backend."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One client (and connection pool) for the whole process, closed on shutdown.
        async with httpx.AsyncClient(
            base_url=str(settings.backend_url),
            timeout=httpx.Timeout(settings.request_timeout_s, connect=settings.connect_timeout_s),
            transport=transport,
        ) as client:
            app.state.backend = client
            yield

    app = FastAPI(title="Relay", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about backends (that is /readyz, Day 4)."""
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body)
        except ValueError:
            return openai_error(400, "Request body must be valid JSON.", "invalid_request_error")
        if not isinstance(payload, dict):
            return openai_error(400, "Request body must be a JSON object.", "invalid_request_error")
        if payload.get("stream") is True:
            # Streaming needs chunk-by-chunk forwarding (Day 2); buffering it would be wrong.
            return openai_error(501, "Streaming is not supported yet.", "not_implemented")

        client: httpx.AsyncClient = request.app.state.backend
        try:
            upstream = await client.post(
                "/v1/chat/completions",
                content=body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.TimeoutException:
            return openai_error(504, "Backend timed out.", "backend_timeout")
        except httpx.RequestError:
            return openai_error(502, "Backend unavailable.", "backend_unavailable")

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    return app


app = create_app()
