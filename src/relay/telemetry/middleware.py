"""Outermost middleware: request id, one access log line and HTTP metrics per request."""

import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from relay.telemetry.logging import access_log
from relay.telemetry.metrics import Metrics

QUIET_PATHS = {"/healthz", "/readyz", "/metrics"}  # probes and scrapes: metrics only, no log


class TelemetryMiddleware:
    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4()
        started = time.perf_counter()
        # Shared with the endpoint through request.state (a plain dict, so values set by
        # the endpoint are visible here even if the response is produced in another task).
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["started"] = started
        fields: dict = state.setdefault("log", {})
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", str(request_id).encode()))
                message = {**message, "headers": headers}
            await send(message)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=str(request_id))
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            duration = time.perf_counter() - started
            route = _route(scope)
            method = scope["method"]
            self.metrics.http_requests.labels(route, method, str(status)).inc()
            self.metrics.http_duration.labels(route, method).observe(duration)
            if scope["path"] not in QUIET_PATHS:
                level = "warning" if status >= 500 else "info"
                getattr(access_log, level)(
                    "request",
                    method=method,
                    path=scope["path"],
                    status=status,
                    duration_ms=round(duration * 1000, 1),
                    **fields,
                )
            structlog.contextvars.clear_contextvars()


def _route(scope: Scope) -> str:
    """The route template (e.g. /v1/models/{model_id}), never the raw path: bounded labels."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if path else "unmatched"
