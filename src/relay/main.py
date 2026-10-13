"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to backends."""

import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from relay.accounting.db import create_pool
from relay.accounting.ledger import UsageLedger
from relay.accounting.usage import UsageEvent
from relay.api import chat, models
from relay.api.errors import install_error_handlers
from relay.api.limits import BodySizeLimitMiddleware
from relay.auth.dependency import require_api_key
from relay.auth.keys import CachedKeyStore, KeyStore, PostgresKeyStore
from relay.backends import ENGINES, Backend, OpenAICompatBackend
from relay.config import PricingConfig, RelayConfig, Settings, load_config, load_pricing
from relay.lifecycle import Lifecycle, LifecycleMiddleware
from relay.limits.failover import FailoverRateLimiter
from relay.limits.token_bucket import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from relay.routing.circuit_breaker import CircuitBreaker
from relay.routing.router import Router
from relay.telemetry.logging import configure_logging
from relay.telemetry.metrics import Metrics, StateCollector
from relay.telemetry.middleware import TelemetryMiddleware
from relay.telemetry.tracing import build_tracer_provider, instrument_app, instrument_backend_client


def build_backends(
    config: RelayConfig, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Backend]:
    """One Backend object (with its own connection pool) per configured backend."""
    return {
        name: ENGINES[cfg.engine](
            name,
            str(cfg.url),
            timeouts=cfg.timeouts(),
            transport=transport,
        )
        for name, cfg in config.backends.items()
    }


def create_app(
    settings: Settings | None = None,
    *,
    config: RelayConfig | None = None,
    pricing: PricingConfig | None = None,
    backends: Mapping[str, Backend] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    usage_sink: Callable[[UsageEvent], None] | None = None,
    key_store: KeyStore | None = None,
    rate_limiter: RateLimiter | None = None,
    tracer_provider: trace.TracerProvider | None = None,
) -> FastAPI:
    """Build the app.

    Config and pricing are loaded and validated here, at startup, so a bad file stops
    the process before it serves anything. Tests can pass a `config`, ready-made fake
    `backends` (no HTTP at all) or a `transport` that replaces the network under real
    backends, and a `key_store` / `rate_limiter` / `usage_sink` instead of Postgres,
    Redis and the usage ledger.
    """
    configure_logging()
    settings = settings or Settings()
    config = config or load_config(settings.config_path)
    pricing = pricing or load_pricing(settings.pricing_path)
    provider = tracer_provider or build_tracer_provider(settings.otel_endpoint)
    tracer = provider.get_tracer("relay")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.pricing = pricing
        app.state.backends = dict(backends) if backends else build_backends(config, transport)
        missing = set(config.backends) - set(app.state.backends)
        if missing:
            raise RuntimeError(f"no backend object for configured backends: {sorted(missing)}")
        app.state.started_at = int(time.time())
        app.state.tracer = tracer
        for backend in app.state.backends.values():
            if isinstance(backend, OpenAICompatBackend):
                instrument_backend_client(backend.http_client, provider)
        # One circuit breaker per backend; the router walks each model's backend list.
        breakers = {name: CircuitBreaker(name, config.breaker.policy()) for name in config.backends}
        app.state.router = Router(
            config,
            app.state.backends,
            breakers,
            config.retry.policy(),
            metrics=metrics,
            tracer=tracer,
        )

        async with AsyncExitStack() as stack:
            if hasattr(provider, "shutdown"):
                stack.callback(provider.shutdown)  # flush buffered spans on exit (runs last)
            for backend in app.state.backends.values():
                stack.push_async_callback(backend.aclose)

            pool = None
            if key_store is None or usage_sink is None:
                # Fail fast: a gateway that cannot authenticate anyone should not start.
                pool = await create_pool(settings.database_url)
                stack.push_async_callback(pool.close)
            app.state.key_store = key_store or CachedKeyStore(PostgresKeyStore(pool))

            if usage_sink is None:
                # Every finished request goes to the ledger (non-blocking queue). It is
                # registered after the pool, so on shutdown it flushes before the pool closes.
                ledger = UsageLedger(pool)
                ledger.start()
                stack.push_async_callback(ledger.close)
                app.state.ledger = ledger
                app.state.usage_sink = ledger.submit
            else:
                app.state.usage_sink = usage_sink

            if rate_limiter is None:
                redis_client = aioredis.from_url(
                    settings.redis_url, socket_connect_timeout=1, socket_timeout=1
                )
                stack.push_async_callback(redis_client.aclose)
                # ADR-002: if Redis is down, enforce the same limits locally.
                app.state.rate_limiter = FailoverRateLimiter(
                    RedisRateLimiter(redis_client), InMemoryRateLimiter()
                )
            else:
                app.state.rate_limiter = rate_limiter
            yield

    app = FastAPI(title="Relay", lifespan=lifespan)
    # Created with the app (not in the lifespan) so the server can flip it on a signal.
    lifecycle = app.state.lifecycle = Lifecycle()
    metrics = app.state.metrics = Metrics()  # one registry per app (tests build many apps)
    metrics.watch(
        StateCollector(
            breakers=lambda: _breaker_states(app),
            in_flight=lambda: lifecycle.in_flight,
            limiter_degraded=lambda: getattr(
                getattr(app.state, "rate_limiter", None), "degraded", None
            ),
            ledger=lambda: _ledger_stats(app),
        )
    )
    install_error_handlers(app)
    # Middleware added later wraps earlier ones, so the order of requests is:
    # Telemetry (request id, access log) -> Lifecycle (draining) -> BodySize -> app.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.limits.max_body_bytes)
    app.add_middleware(LifecycleMiddleware, lifecycle=lifecycle)
    app.add_middleware(TelemetryMiddleware, metrics=metrics)
    # Instrumented last so its server span is outermost: our access log line then
    # carries the trace id.
    instrument_app(app, provider)
    # Every API route requires an API key; /healthz stays open for load balancers.
    authenticated = [Depends(require_api_key)]
    app.include_router(chat.router, dependencies=authenticated)
    app.include_router(models.router, dependencies=authenticated)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up (also while draining). If not: restart it."""
        return {"status": "ok"}

    @app.get("/readyz", responses={503: {"description": "Draining: send no new traffic"}})
    async def readyz() -> JSONResponse:
        """Readiness: should this instance receive new traffic? 503 once shutdown starts."""
        body = {
            "status": "draining" if lifecycle.draining else "ready",
            "in_flight": lifecycle.in_flight,
            "breakers": _breaker_states(app),
        }
        return JSONResponse(body, status_code=503 if lifecycle.draining else 200)

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Prometheus scrape endpoint (bind Relay to an internal network in production)."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _breaker_states(app: FastAPI) -> dict[str, str]:
    router = getattr(app.state, "router", None)
    return {n: b.state.value for n, b in router.breakers.items()} if router else {}


def _ledger_stats(app: FastAPI) -> dict[str, int] | None:
    ledger = getattr(app.state, "ledger", None)
    if ledger is None:
        return None
    return {
        "queued": ledger.queued,
        "written": ledger.written,
        "dropped": ledger.dropped,
        "rejected": ledger.rejected,
    }


app = create_app()
