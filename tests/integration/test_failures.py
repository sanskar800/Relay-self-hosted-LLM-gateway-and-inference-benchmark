"""Failure scenarios end to end: real sockets, real timeouts, real broken connections.

Backends are scripted engines served by uvicorn on local ports (tests/engine_server.py),
so these tests need no Docker; they are integration tests because they exercise the
real network path (httpx, timeouts, connection resets).
"""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import redis.asyncio as aioredis

from relay.accounting.ledger import UsageLedger
from relay.config import RelayConfig
from relay.limits.failover import FailoverRateLimiter
from relay.limits.token_bucket import InMemoryRateLimiter, RedisRateLimiter
from tests.engine_server import serve_engine
from tests.fake_engine import FakeEngine
from tests.unit.fakes import TEST_KEY, StaticKeyStore, make_app, make_client, make_principal

pytestmark = pytest.mark.integration

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}


def config(backends: dict[str, str], **backend_opts) -> RelayConfig:
    return RelayConfig.model_validate(
        {
            "backends": {
                name: {"engine": "llamacpp", "url": url, **backend_opts}
                for name, url in backends.items()
            },
            "models": {"m": {"backends": list(backends)}},
            "retry": {"base_delay_s": 0},
        }
    )


@contextmanager
def relay(cfg: RelayConfig, **app_kwargs) -> Iterator:
    with make_client(make_app(config=cfg, **app_kwargs)) as client:
        yield client


def events_of(content: bytes) -> list[bytes]:
    return [e for e in content.split(b"\n\n") if e]


# 1. Backend killed mid-stream ------------------------------------------------------------


def test_backend_crash_mid_stream_ends_cleanly_and_is_not_retried() -> None:
    engine = FakeEngine(tokens=[f"t{i}" for i in range(10)])
    with serve_engine(engine, crash_after_tokens=3) as url, relay(config({"b": url})) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
        breaker = client.app.state.router.breakers["b"]
        failures = breaker.consecutive_failures
    events = events_of(resp.content)
    assert resp.status_code == 200
    assert sum(b'"content": "t' in e for e in events) == 3  # what the client already had
    error = json.loads(events[-1].removeprefix(b"data: "))["error"]
    assert error["type"] == "backend_unavailable"  # a clean, parseable ending
    assert b"[DONE]" not in resp.content
    assert len(engine.requests) == 1  # never retried after tokens were sent
    assert failures == 1  # but it counts against the backend's breaker


# 2. Slow backends ------------------------------------------------------------------------


def test_backend_too_slow_to_start_falls_back_to_the_next() -> None:
    slow, fast = FakeEngine(first_event_delay_s=3.0), FakeEngine(tokens=["ok"])
    with (
        serve_engine(slow) as slow_url,
        serve_engine(fast) as fast_url,
        relay(
            config({"primary": slow_url, "secondary": fast_url}, first_token_timeout_s=0.3)
        ) as client,
    ):
        start = time.perf_counter()
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
        elapsed = time.perf_counter() - start
    assert resp.status_code == 200
    assert resp.headers["x-relay-backend"] == "secondary"
    assert b"[DONE]" in resp.content
    assert len(slow.requests) == 1  # a timeout is not retried on the same backend
    assert elapsed < 2.5  # did not wait for the slow backend


def test_backend_going_silent_mid_stream_hits_the_idle_timeout() -> None:
    # Real socket: this is httpx's own read timeout, which an in-memory transport
    # cannot trigger.
    engine = FakeEngine(tokens=list("abcdef"), stall_after=2, stall_s=5.0)
    with serve_engine(engine) as url, relay(config({"b": url}, idle_timeout_s=0.4)) as client:
        start = time.perf_counter()
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
        elapsed = time.perf_counter() - start
    events = events_of(resp.content)
    assert sum(b'"content": "' in e for e in events) == 2
    assert json.loads(events[-1].removeprefix(b"data: "))["error"]["type"] == "backend_timeout"
    assert elapsed < 3  # cut after ~0.4 s of silence, not after the 5 s stall


# 3. Redis down ---------------------------------------------------------------------------


def test_redis_down_requests_still_served_and_still_limited() -> None:
    dead_redis = aioredis.from_url(
        "redis://127.0.0.1:6390/0", socket_connect_timeout=0.5, socket_timeout=0.5
    )
    limiter = FailoverRateLimiter(RedisRateLimiter(dead_redis), InMemoryRateLimiter())
    keys = StaticKeyStore({TEST_KEY: make_principal(requests_per_minute=2)})
    with (
        serve_engine(FakeEngine()) as url,
        relay(config({"b": url}), rate_limiter=limiter, key_store=keys) as client,
    ):
        statuses = [client.post("/v1/chat/completions", json=BODY).status_code for _ in range(3)]
        degraded = client.get("/metrics").text
    assert statuses == [200, 200, 429]  # ADR-002: fail open, but to local limits
    assert "relay_ratelimiter_degraded 1.0" in degraded


# 4. Postgres down ------------------------------------------------------------------------


class DeadPool:
    """A pool whose database is gone: every acquire is refused."""

    def acquire(self):
        raise ConnectionRefusedError("database is down")


def test_postgres_down_requests_still_served_and_usage_buffered() -> None:
    ledger = UsageLedger(DeadPool(), flush_interval_s=0.05, initial_backoff_s=0.05)
    with (
        serve_engine(FakeEngine()) as url,
        relay(config({"b": url}), usage_sink=ledger.submit) as client,
    ):
        # The ledger's writer must run on the app's event loop (TestClient's own thread).
        client.portal.call(ledger.start)
        statuses = [client.post("/v1/chat/completions", json=BODY).status_code for _ in range(5)]
        time.sleep(0.5)  # give the writer time to try (and fail)
        failures, buffered = ledger.write_failures, ledger.queued + len(ledger._inflight)
        client.portal.call(ledger.close, 0.2)  # stop the writer (its final flush fails too)
    assert statuses == [200] * 5  # the ledger is never in the request path
    assert failures >= 1
    assert buffered == 5  # buffered, not lost
    assert ledger.dropped == 0
