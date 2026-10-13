"""Readiness and graceful shutdown.

When shutdown starts, Relay stops being *ready* (so load balancers stop sending new
traffic) but stays *alive* while in-flight requests finish. New API requests during
that window get 503 with Retry-After, which clients retry elsewhere.
"""

import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

from relay.api.errors import openai_error

log = logging.getLogger("relay.lifecycle")


class Lifecycle:
    def __init__(self) -> None:
        self.draining = False
        self.draining_since: float | None = None
        self.in_flight = 0  # API requests currently being served (incl. open streams)

    def start_draining(self) -> None:
        if not self.draining:
            self.draining = True
            self.draining_since = time.monotonic()
            log.warning("shutdown started: not ready; %d request(s) in flight", self.in_flight)


class LifecycleMiddleware:
    """Counts in-flight API requests and refuses new ones while draining."""

    def __init__(self, app: ASGIApp, lifecycle: Lifecycle) -> None:
        self.app = app
        self.lifecycle = lifecycle

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            await self.app(scope, receive, send)  # /healthz, /readyz, /docs: always served
            return
        if self.lifecycle.draining:
            response = openai_error(
                503,
                "Relay is shutting down; retry the request.",
                "service_unavailable",
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return
        # Counted until the response is fully sent: for streams, until the stream ends.
        self.lifecycle.in_flight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.lifecycle.in_flight -= 1
