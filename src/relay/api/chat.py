"""POST /v1/chat/completions: validated, then proxied (buffered or streamed) to the backend."""

import json
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from relay.api.errors import openai_error
from relay.api.schemas import ChatCompletionRequest, ErrorResponse

router = APIRouter()

UPSTREAM_PATH = "/v1/chat/completions"
JSON_HEADERS = {"Content-Type": "application/json"}
SSE_HEADERS = {
    # Never cache a token stream; ask any reverse proxy in front not to buffer it.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


@router.post(
    "/v1/chat/completions",
    responses={status: {"model": ErrorResponse} for status in (400, 502, 504)},
)
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
    # `payload` is validated (invalid -> 400 via the error handler); the backend still
    # receives the ORIGINAL bytes, so unknown fields and formatting are untouched.
    body = await request.body()
    client: httpx.AsyncClient = request.app.state.backend
    if payload.stream:
        return await _stream(client, body)
    return await _complete(client, body)


async def _complete(client: httpx.AsyncClient, body: bytes) -> Response:
    try:
        upstream = await client.post(UPSTREAM_PATH, content=body, headers=JSON_HEADERS)
    except httpx.TimeoutException:
        return openai_error(504, "Backend timed out.", "backend_timeout")
    except httpx.RequestError:
        return openai_error(502, "Backend unavailable.", "backend_unavailable")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


async def _stream(client: httpx.AsyncClient, body: bytes) -> Response:
    # Phase 1, before any byte reaches the client: we can still answer with a normal
    # HTTP error status, because no response headers have been sent yet.
    request = client.build_request("POST", UPSTREAM_PATH, content=body, headers=JSON_HEADERS)
    try:
        upstream = await client.send(request, stream=True)
    except httpx.TimeoutException:
        return openai_error(504, "Backend timed out.", "backend_timeout")
    except httpx.RequestError:
        return openai_error(502, "Backend unavailable.", "backend_unavailable")

    if upstream.status_code != 200:
        # The backend refused the request (e.g. 400): pass its error through unchanged.
        content = await upstream.aread()
        await upstream.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    # Phase 2: 200 + headers go out with the first chunk. From here on the status code
    # cannot change, so failures become an SSE error event.
    return StreamingResponse(
        _relay_chunks(upstream), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _relay_chunks(upstream: httpx.Response) -> AsyncIterator[bytes]:
    """Forward each SSE event as soon as it is complete.

    Pulling the next upstream chunk only after each yield gives backpressure: a slow
    client slows the backend instead of growing a buffer here.
    """
    try:
        async for event in iter_sse_events(upstream.aiter_bytes()):
            yield event
    except httpx.TimeoutException:
        yield _sse_error("Backend stopped sending tokens (timeout).", "backend_timeout")
    except httpx.TransportError:
        yield _sse_error("Backend connection lost mid-stream.", "backend_unavailable")
    finally:
        # Runs on normal end, on upstream failure, and when the client disconnects
        # (Starlette cancels this generator). Closing the upstream response closes the
        # connection to the engine, which stops generating for this request.
        await upstream.aclose()


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Re-cut a byte stream into whole SSE events (each ending in a blank line).

    TCP chunks can split an event anywhere. Forwarding whole events means a mid-stream
    failure never leaves half an event in front of the error event we send.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk.replace(b"\r\n", b"\n")
        while (end := buffer.find(b"\n\n")) != -1:
            yield buffer[: end + 2]
            buffer = buffer[end + 2 :]
    if buffer.strip():
        # Stream ended without the final blank line: still deliver the last event.
        yield buffer.rstrip(b"\n") + b"\n\n"


def _sse_error(message: str, error_type: str) -> bytes:
    error = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return f"data: {json.dumps(error)}\n\n".encode()
