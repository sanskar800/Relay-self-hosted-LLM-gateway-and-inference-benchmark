"""Relay gateway: FastAPI app that proxies OpenAI-compatible requests to backends."""

import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI

from relay.accounting.db import create_pool
from relay.accounting.ledger import UsageLedger
from relay.accounting.usage import UsageEvent
from relay.api import chat, models
from relay.api.errors import install_error_handlers
from relay.api.limits import BodySizeLimitMiddleware
from relay.auth.dependency import require_api_key
from relay.auth.keys import CachedKeyStore, KeyStore, PostgresKeyStore
from relay.backends import ENGINES, Backend
from relay.config import PricingConfig, RelayConfig, Settings, load_config, load_pricing
from relay.limits.failover import FailoverRateLimiter
from relay.limits.token_bucket import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from relay.routing.circuit_breaker import CircuitBreaker
from relay.routing.router import Router


def configure_logging() -> None:
    """Minimal console logging for Relay's own loggers (structured JSON logs come later).

    uvicorn configures only its own loggers, so without this our INFO lines are dropped.
    """
    logger = logging.getLogger("relay")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


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

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.config = config
        app.state.pricing = pricing
        app.state.backends = dict(backends) if backends else build_backends(config, transport)
        missing = set(config.backends) - set(app.state.backends)
        if missing:
            raise RuntimeError(f"no backend object for configured backends: {sorted(missing)}")
        app.state.started_at = int(time.time())
        # One circuit breaker per backend; the router walks each model's backend list.
        breakers = {name: CircuitBreaker(name, config.breaker.policy()) for name in config.backends}
        app.state.router = Router(config, app.state.backends, breakers, config.retry.policy())

        async with AsyncExitStack() as stack:
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
    install_error_handlers(app)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=config.limits.max_body_bytes)
    # Every API route requires an API key; /healthz stays open for load balancers.
    authenticated = [Depends(require_api_key)]
    app.include_router(chat.router, dependencies=authenticated)
    app.include_router(models.router, dependencies=authenticated)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up. Says nothing about backends (that is /readyz, Day 4)."""
        return {"status": "ok"}

    return app


app = create_app()
