"""POST /v1/chat/completions: validated, then proxied (buffered or streamed) to a backend."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from relay.accounting.usage import (
    StreamUsage,
    UsageRecord,
    request_stream_usage,
    usage_from_completion,
)
from relay.api.errors import model_not_found, openai_error
from relay.api.schemas import ChatCompletionRequest, ErrorResponse
from relay.backends.base import (
    Backend,
    BackendHTTPError,
    BackendStream,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)

router = APIRouter()

SSE_HEADERS = {
    # Never cache a token stream; ask any reverse proxy in front not to buffer it.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


@router.post(
    "/v1/chat/completions",
    responses={status: {"model": ErrorResponse} for status in (400, 404, 502, 504)},
)
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
    # `payload` is validated (invalid -> 400 via the error handler). The backend receives
    # the client's ORIGINAL bytes, except that streams may get include_usage added.
    body = await request.body()
    route = request.app.state.config.models.get(payload.model)
    if route is None:
        return model_not_found(payload.model)
    # First backend in preference order (fallback down the list comes with routing).
    backend: Backend = request.app.state.backends[route.backends[0]]
    sink: Callable[[UsageRecord], None] = request.app.state.usage_sink

    def record(status: int, usage: tuple) -> None:
        sink(UsageRecord(payload.model, backend.name, payload.stream, status, *usage))

    try:
        if payload.stream:
            options = payload.stream_options
            client_wants_usage = bool(options and options.include_usage)
            upstream_body = body if client_wants_usage else request_stream_usage(body)
            # Phase 1: nothing has been sent to the client yet, so a failure here can
            # still be an ordinary HTTP error response.
            upstream = await backend.stream(upstream_body)
        else:
            result = await backend.chat(body)
            record(result.status_code, usage_from_completion(result.content))
            return Response(result.content, result.status_code, media_type=result.content_type)
    except BackendHTTPError as exc:
        # The backend refused the request (e.g. 400): pass its error through unchanged.
        return Response(exc.content, exc.status_code, media_type=exc.content_type)
    except BackendTimeout:
        return openai_error(504, "Backend timed out.", "backend_timeout")
    except BackendUnavailable:
        return openai_error(502, "Backend unavailable.", "backend_unavailable")

    # Phase 2: 200 + headers go out with the first event. From here on the status code
    # cannot change, so failures become an SSE error event.
    usage = StreamUsage(forward_usage_chunk=client_wants_usage)
    return StreamingResponse(
        _relay_events(upstream, usage, lambda: record(200, usage.result())),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def _relay_events(
    upstream: BackendStream, usage: StreamUsage, on_finish: Callable[[], None]
) -> AsyncIterator[bytes]:
    """Forward each event as soon as it is complete, noting token usage on the way.

    Pulling the next event only after each yield gives backpressure: a slow client slows
    the backend instead of growing a buffer here.
    """
    try:
        async with aclosing(upstream.events()) as events:
            async for event in events:
                if usage.observe(event):
                    yield event
    except BackendStreamError as exc:
        if exc.timeout:
            yield _sse_error("Backend stopped sending tokens (timeout).", "backend_timeout")
        else:
            yield _sse_error("Backend connection lost mid-stream.", "backend_unavailable")
    finally:
        # Runs on normal end, on backend failure, and when the client disconnects.
        # Record first (synchronous, cannot be interrupted), then close the backend
        # stream, which makes the engine stop generating. The close is shielded: after a
        # disconnect this code runs inside a cancelled scope, where an unshielded await
        # could be cancelled before the connection is released.
        on_finish()
        with anyio.CancelScope(shield=True):
            await upstream.aclose()


def _sse_error(message: str, error_type: str) -> bytes:
    error = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return f"data: {json.dumps(error)}\n\n".encode()
