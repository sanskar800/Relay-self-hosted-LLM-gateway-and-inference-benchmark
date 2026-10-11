"""POST /v1/chat/completions: validated, then proxied (buffered or streamed) to a backend."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from relay.accounting.usage import StreamUsage, UsageRecord, UsageSource, usage_from_completion
from relay.api.errors import backend_error_response, openai_error
from relay.api.limits import check_request_limits
from relay.api.schemas import ChatCompletionRequest, ErrorResponse
from relay.api.upstream import upstream_body
from relay.auth.keys import Principal
from relay.auth.policy import check_max_tokens, check_model, default_max_tokens
from relay.backends.base import (
    Backend,
    BackendHTTPError,
    BackendStream,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.limits.estimate import estimate_prompt_tokens, reservation
from relay.limits.responses import rate_limited_error, ratelimit_headers
from relay.limits.token_bucket import RateLimiter, RateLimiterUnavailable

log = logging.getLogger("relay.limits")
router = APIRouter()

SSE_HEADERS = {
    # Never cache a token stream; ask any reverse proxy in front not to buffer it.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

Usage = tuple[int | None, int | None, UsageSource]


@router.post(
    "/v1/chat/completions",
    responses={status: {"model": ErrorResponse} for status in (400, 404, 413, 429, 502, 504)},
)
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
    # `payload` is validated (invalid -> 400 via the error handler). The backend receives
    # the client's ORIGINAL bytes apart from the additions documented in upstream.py.
    body = await request.body()
    config = request.app.state.config
    principal: Principal = request.state.principal
    if limit_error := check_request_limits(payload, config.limits):
        return limit_error
    # Tenant policy: unknown and forbidden models get the same 404; then the token cap.
    check_model(payload.model, principal, config)
    check_max_tokens(payload, principal)

    # Rate limit: reserve 1 request + (estimated prompt + max completion) tokens now,
    # settle against the real usage when the request is over.
    cap = default_max_tokens(payload, principal)
    reserved = reservation(payload, cap or config.limits.max_tokens)
    limiter: RateLimiter = request.app.state.rate_limiter
    decision = await limiter.acquire(principal, reserved)
    if not decision.allowed:
        raise rate_limited_error(decision, reserved)
    rl_headers = ratelimit_headers(decision)

    # First backend in preference order (fallback down the list comes with routing).
    backend: Backend = request.app.state.backends[config.models[payload.model].backends[0]]
    sink: Callable[[UsageRecord], None] = request.app.state.usage_sink

    async def finish(status: int, usage: Usage | None) -> None:
        """Record usage (if any) and give back the unused part of the reservation."""
        if usage is not None:
            sink(UsageRecord(payload.model, backend.name, payload.stream, status, *usage))
        await _settle(limiter, principal, reserved, _actual_tokens(payload, usage))

    options = payload.stream_options
    client_wants_usage = bool(options and options.include_usage)
    outgoing = upstream_body(
        body, include_usage=payload.stream and not client_wants_usage, default_max_tokens=cap
    )
    try:
        if payload.stream:
            # Phase 1: nothing has been sent to the client yet, so a failure here can
            # still be an ordinary HTTP error response.
            upstream = await backend.stream(outgoing)
        else:
            result = await backend.chat(outgoing)
            await finish(result.status_code, usage_from_completion(result.content))
            return Response(
                result.content,
                result.status_code,
                media_type=result.content_type,
                headers=rl_headers,
            )
    except (BackendHTTPError, BackendTimeout, BackendUnavailable) as exc:
        await finish(0, None)  # no tokens were generated: refund the whole reservation
        if isinstance(exc, BackendHTTPError):
            # The backend refused the request (e.g. 400): pass its error through.
            return backend_error_response(exc)
        if isinstance(exc, BackendTimeout):
            return openai_error(504, "Backend timed out.", "backend_timeout")
        return openai_error(502, "Backend unavailable.", "backend_unavailable")

    # Phase 2: 200 + headers go out with the first event. From here on the status code
    # cannot change, so failures become an SSE error event.
    usage = StreamUsage(forward_usage_chunk=client_wants_usage)
    return StreamingResponse(
        _relay_events(upstream, usage, lambda: finish(200, usage.result())),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, **rl_headers},
    )


def _actual_tokens(payload: ChatCompletionRequest, usage: Usage | None) -> int | None:
    """Tokens the request really used; None if unknown (then the reservation stands)."""
    if usage is None:
        return 0
    prompt, completion, source = usage
    if source == "missing" or completion is None:
        return None
    return (prompt if prompt is not None else estimate_prompt_tokens(payload)) + completion


async def _settle(
    limiter: RateLimiter, principal: Principal, reserved: int, actual: int | None
) -> None:
    if actual is None:
        return
    try:
        await limiter.settle(principal, reserved - actual)
    except RateLimiterUnavailable as exc:
        # Best effort: a failed refund only makes the bucket stricter for a moment.
        log.warning("could not settle rate-limit reservation: %s", exc)


async def _relay_events(
    upstream: BackendStream, usage: StreamUsage, on_finish: Callable[[], Awaitable[None]]
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
        # Runs on normal end, on backend failure, and when the client disconnects. After
        # a disconnect this code runs inside a cancelled scope, where an unshielded await
        # could be cancelled. So both steps are shielded: first close the backend stream
        # (the engine stops generating), then record usage and settle the reservation.
        with anyio.CancelScope(shield=True):
            await upstream.aclose()
            await on_finish()


def _sse_error(message: str, error_type: str) -> bytes:
    error = {"error": {"message": message, "type": error_type, "param": None, "code": None}}
    return f"data: {json.dumps(error)}\n\n".encode()
