"""Routing with fallback: try a model's backends in order of preference.

For each backend: skip it if its circuit breaker is open; otherwise call it with retries
(ADR-003). Unavailable, timed-out or 5xx backends fall back to the next one; a 4xx is
returned at once, because every backend would reject the same request.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping

from relay.backends.base import (
    Backend,
    BackendError,
    BackendHTTPError,
    BackendResponse,
    BackendStream,
    BackendStreamError,
    BackendTimeout,
    BackendUnavailable,
)
from relay.config import RelayConfig
from relay.routing.circuit_breaker import CircuitBreaker
from relay.routing.retry import RetryPolicy, call_with_retry

log = logging.getLogger("relay.routing")


class CircuitOpen(BackendError):
    """The backend was skipped because its breaker is open (no request was sent)."""


class NoBackendAvailable(BackendError):
    """Every backend for the model was skipped by its circuit breaker."""


def is_backend_failure(exc: BackendError) -> bool:
    """Says something about the backend's health (counts towards its breaker)."""
    if isinstance(exc, BackendHTTPError):
        return exc.status_code >= 500
    return isinstance(exc, (BackendUnavailable, BackendTimeout, BackendStreamError))


class Router:
    def __init__(
        self,
        config: RelayConfig,
        backends: Mapping[str, Backend],
        breakers: Mapping[str, CircuitBreaker],
        retry: RetryPolicy,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._backends = backends
        self.breakers = breakers
        self._retry = retry
        self._sleep = sleep

    async def chat(self, model: str, body: bytes) -> tuple[Backend, BackendResponse]:
        return await self._route(model, lambda backend: backend.chat(body))

    async def stream(self, model: str, body: bytes) -> tuple[Backend, BackendStream]:
        return await self._route(model, lambda backend: backend.stream(body))

    async def _route[T](
        self, model: str, call: Callable[[Backend], Awaitable[T]]
    ) -> tuple[Backend, T]:
        last_error: BackendError | None = None
        for name in self._config.models[model].backends:
            backend, breaker = self._backends[name], self.breakers[name]
            failures: list[BackendError] = []  # real errors from this backend

            async def attempt(
                backend: Backend = backend,
                breaker: CircuitBreaker = breaker,
                failures: list[BackendError] = failures,
            ) -> T:
                # Checked before every attempt, so a breaker that opens during this
                # request's own retries also stops the remaining retries.
                if not breaker.allow_request():
                    raise CircuitOpen(backend.name, "circuit open")
                try:
                    result = await call(backend)
                except BackendError as exc:
                    if is_backend_failure(exc):
                        breaker.record_failure()
                        failures.append(exc)
                    else:
                        breaker.record_success()  # it answered properly (e.g. a 4xx)
                    raise
                breaker.record_success()
                return result

            try:
                return backend, await call_with_retry(attempt, self._retry, sleep=self._sleep)
            except CircuitOpen:
                # Skipped without sending anything, or opened during our own retries:
                # then report the real failure, not "circuit open".
                if failures:
                    last_error = failures[-1]
                continue
            except BackendError as exc:
                if not is_backend_failure(exc):
                    raise  # e.g. 400: every backend would reject it
                last_error = exc
                log.warning("backend %s failed (%s); trying the next backend for %s",
                            name, exc, model)  # fmt: skip
        if last_error is not None:
            raise last_error
        raise NoBackendAvailable(model, "every backend's circuit breaker is open")
