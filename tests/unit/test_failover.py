"""ADR-002: when Redis is down, limits are enforced per instance instead of failing."""

import logging

from relay.limits.failover import FailoverRateLimiter
from relay.limits.token_bucket import Decision, InMemoryRateLimiter, RateLimiterUnavailable
from tests.unit.fakes import (
    TEST_KEY,
    FakeBackend,
    StaticKeyStore,
    make_app,
    make_client,
    make_config,
    make_principal,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class SwitchableLimiter(InMemoryRateLimiter):
    """Stands in for Redis: works like the real limiter until switched off."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.up = True
        self.calls = 0

    async def acquire(self, principal, tokens: int) -> Decision:
        self.calls += 1
        if not self.up:
            raise RateLimiterUnavailable("connection refused")
        return await super().acquire(principal, tokens)

    async def settle(self, principal, refund: int) -> None:
        if not self.up:
            raise RateLimiterUnavailable("connection refused")
        await super().settle(principal, refund)


def setup(retry_interval_s: float = 5.0):
    clock = FakeClock()
    primary = SwitchableLimiter(clock)
    limiter = FailoverRateLimiter(
        primary, InMemoryRateLimiter(clock), retry_interval_s=retry_interval_s, clock=clock
    )
    return clock, primary, limiter


async def test_redis_down_falls_back_to_local_limits_that_still_bound_traffic() -> None:
    _, primary, limiter = setup()
    primary.up = False
    p = make_principal(requests_per_minute=2)
    results = [(await limiter.acquire(p, 1)).allowed for _ in range(3)]
    assert results == [True, True, False]  # fail open, but not unlimited
    assert limiter.degraded and limiter.fallback_decisions == 3


async def test_dead_redis_is_skipped_for_the_retry_interval(caplog) -> None:
    clock, primary, limiter = setup(retry_interval_s=5)
    primary.up = False
    p = make_principal()
    with caplog.at_level(logging.INFO, logger="relay.limits"):
        for _ in range(10):
            await limiter.acquire(p, 1)
        assert primary.calls == 1  # not one timeout per request
        clock.now += 5
        primary.up = True
        await limiter.acquire(p, 1)
    assert primary.calls == 2 and not limiter.degraded
    # One warning when degrading and one info when recovering, not one per request.
    assert caplog.text.count("Redis unavailable") == 1
    assert caplog.text.count("shared limits restored") == 1


async def test_settle_goes_where_limits_are_enforced() -> None:
    _, primary, limiter = setup()
    p = make_principal(requests_per_minute=100, tokens_per_minute=100)
    primary.up = False
    await limiter.acquire(p, 90)
    await limiter.settle(p, 80)  # Redis still down: refunded to the local bucket
    assert (await limiter.acquire(p, 85)).allowed


def test_gateway_keeps_serving_with_redis_down() -> None:
    clock = FakeClock()
    primary = SwitchableLimiter(clock)
    primary.up = False
    limiter = FailoverRateLimiter(primary, InMemoryRateLimiter(clock), clock=clock)
    backend = FakeBackend("b")
    app = make_app(
        config=make_config("m", "b"),
        backends={"b": backend},
        key_store=StaticKeyStore({TEST_KEY: make_principal(requests_per_minute=2)}),
        rate_limiter=limiter,
    )
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    with make_client(app) as client:
        statuses = [client.post("/v1/chat/completions", json=body).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert len(backend.requests) == 2
