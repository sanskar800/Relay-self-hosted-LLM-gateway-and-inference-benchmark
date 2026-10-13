"""POST /v1/chat/completions: validated, then proxied (buffered or streamed) to a backend."""

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from relay.accounting.usage import (
    StreamUsage,
    UsageEvent,
    UsageRecord,
    UsageSource,
    usage_from_completion,
)
from relay.api.errors import backend_error_response, openai_error
from relay.api.limits import check_request_limits
from relay.api.schemas import ChatCompletionRequest, ErrorResponse
from relay.api.upstream import upstream_body
from relay.auth.keys import Principal
from relay.auth.policy import check_max_tokens, check_model, default_max_tokens
from relay.backends.base import (
    BackendError,
    BackendHTTPError,
    BackendStream,
    BackendStreamError,
    BackendTimeout,
)
from relay.config import PricingConfig
from relay.limits.estimate import estimate_prompt_tokens, reservation
from relay.limits.responses import rate_limited_error, ratelimit_headers
from relay.limits.token_bucket import RateLimiter, RateLimiterUnavailable
from relay.routing.router import NoBackendAvailable, Router
from relay.telemetry.metrics import Metrics

log = logging.getLogger("relay.limits")
router = APIRouter()

SSE_HEADERS = {
    # Never cache a token stream; ask any reverse proxy in front not to buffer it.
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

Usage = tuple[int | None, int | None, UsageSource]
MISSING: Usage = (None, None, "missing")  # a request that produced no usage report


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
    metrics: Metrics = request.app.state.metrics
    # Fields for this request's access log line (written by TelemetryMiddleware).
    log_fields: dict = request.state.log
    log_fields.update(
        tenant=principal.tenant_id,
        key=principal.key_prefix,
        model=payload.model,
        stream=payload.stream,
    )
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
        if decision.too_large:
            kind = "too_large"
        else:
            kind = "requests" if decision.remaining_requests < 1 else "tokens"
        metrics.rate_limited.labels(principal.tenant_id, kind).inc()
        raise rate_limited_error(decision, reserved)
    headers = ratelimit_headers(decision)

    router: Router = request.app.state.router
    sink: Callable[[UsageEvent], None] = request.app.state.usage_sink
    pricing: PricingConfig = request.app.state.pricing
    # Set by TelemetryMiddleware at the very start of the request (before auth).
    request_id, started = request.state.request_id, request.state.started
    occurred_at = datetime.now(UTC)
    # Which backend served (or last failed) the request; for the ledger and a header.
    served_by = config.models[payload.model].backends[0]
    outcome = {"stream_error": False}

    async def finish(status: int, usage: Usage | None, stream: StreamUsage | None) -> None:
        """Ledger event, metrics and log fields; then settle the rate-limit reservation."""
        record = UsageRecord(payload.model, served_by, payload.stream, status, *(usage or MISSING))
        first = stream.first_content_at if stream else None
        event = UsageEvent(
            request_id=request_id,
            occurred_at=occurred_at,
            tenant_id=principal.tenant_id,
            api_key_id=principal.key_id,
            usage=record,
            ttft_ms=_ms(first - started) if first else None,
            duration_ms=_ms(time.perf_counter() - started),
            cost_gbp=pricing.cost_gbp(
                payload.model, record.prompt_tokens, record.completion_tokens
            ),
        )
        sink(event)
        _observe(metrics, event, stream, outcome["stream_error"])
        log_fields.update(
            backend=served_by,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            usage_source=record.source,
            ttft_ms=event.ttft_ms,
        )
        await _settle(limiter, principal, reserved, _actual_tokens(payload, usage))

    options = payload.stream_options
    client_wants_usage = bool(options and options.include_usage)
    outgoing = upstream_body(
        body, include_usage=payload.stream and not client_wants_usage, default_max_tokens=cap
    )
    # Relay's own time so far (auth, validation, policy, rate limit): gateway overhead.
    metrics.gateway_overhead.observe(time.perf_counter() - started)
    try:
        if payload.stream:
            # Phase 1: the router tries the model's backends in order (skipping open
            # breakers, retrying per ADR-003) until one produces its first event. Until
            # then nothing has reached the client; failures become an HTTP error.
            backend, upstream = await router.stream(payload.model, outgoing)
        else:
            # Nothing is sent to the client before the whole answer exists.
            backend, result = await router.chat(payload.model, outgoing)
            served_by = backend.name
            await finish(result.status_code, usage_from_completion(result.content), None)
            return Response(
                result.content,
                result.status_code,
                media_type=result.content_type,
                headers={**headers, "x-relay-backend": backend.name},
            )
    except BackendError as exc:
        served_by = exc.backend if exc.backend in config.backends else served_by
        if isinstance(exc, BackendHTTPError):
            # The backend refused the request (e.g. 400): pass its error through.
            response = backend_error_response(exc)
        elif isinstance(exc, NoBackendAvailable):
            response = openai_error(
                503, "No backend is available for this model right now.", "service_unavailable"
            )
        elif isinstance(exc, BackendTimeout):
            response = openai_error(504, "Backend timed out.", "backend_timeout")
        else:
            response = openai_error(502, "Backend unavailable.", "backend_unavailable")
        # Recorded for the audit trail; no tokens were generated, so all are refunded.
        await finish(response.status_code, None, None)
        return response

    # Phase 2: 200 + headers go out with the first event. From here on the status code
    # cannot change, so failures become an SSE error event (and count against the
    # backend's breaker), never a retry or a fallback.
    served_by = backend.name
    usage = StreamUsage(forward_usage_chunk=client_wants_usage)
    metrics.backend_in_flight.labels(served_by).inc()

    def stream_failed() -> None:
        outcome["stream_error"] = True
        router.breakers[served_by].record_failure()

    async def stream_finished() -> None:
        metrics.backend_in_flight.labels(served_by).dec()
        await finish(200, usage.result(), usage)

    return StreamingResponse(
        _relay_events(upstream, usage, stream_finished, on_stream_error=stream_failed),
        media_type="text/event-stream",
        headers={**SSE_HEADERS, **headers, "x-relay-backend": served_by},
    )


def _observe(
    metrics: Metrics, event: UsageEvent, stream: StreamUsage | None, stream_error: bool
) -> None:
    u = event.usage
    if stream_error:
        outcome = "stream_error"
    else:
        outcome = "ok" if u.status_code < 400 else f"{u.status_code // 100}xx"
    metrics.backend_requests.labels(u.backend, outcome).inc()
    if event.ttft_ms is not None:
        metrics.ttft.labels(u.model, u.backend).observe(event.ttft_ms / 1000)
    if stream is not None:
        itl = metrics.itl.labels(u.model, u.backend)
        for gap in stream.token_gaps:
            itl.observe(gap)
    if u.prompt_tokens is not None:
        metrics.prompt_tokens.labels(event.tenant_id, u.model).inc(u.prompt_tokens)
    if u.completion_tokens is not None:
        metrics.completion_tokens.labels(event.tenant_id, u.model).inc(u.completion_tokens)
    if event.cost_gbp is not None:
        metrics.cost_gbp.labels(event.tenant_id, u.model).inc(event.cost_gbp)


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


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
    upstream: BackendStream,
    usage: StreamUsage,
    on_finish: Callable[[], Awaitable[None]],
    on_stream_error: Callable[[], None] = lambda: None,
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
        on_stream_error()  # the backend broke mid-answer: counts against its breaker
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
