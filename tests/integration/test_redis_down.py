"""ADR-002 with a real Redis client against an address where nothing listens."""

import time

import pytest
import redis.asyncio as aioredis

from relay.limits.failover import FailoverRateLimiter
from relay.limits.token_bucket import InMemoryRateLimiter, RedisRateLimiter
from tests.unit.fakes import make_principal

pytestmark = pytest.mark.integration

DEAD_REDIS = "redis://127.0.0.1:6390/0"  # nothing listens here


async def test_unreachable_redis_degrades_to_local_limits_without_slowing_requests() -> None:
    client = aioredis.from_url(DEAD_REDIS, socket_connect_timeout=1, socket_timeout=1)
    limiter = FailoverRateLimiter(RedisRateLimiter(client), InMemoryRateLimiter())
    p = make_principal(requests_per_minute=3)
    try:
        start = time.perf_counter()
        first = await limiter.acquire(p, 1)  # discovers Redis is down
        after_first = time.perf_counter()
        rest = [await limiter.acquire(p, 1) for _ in range(3)]
        after_rest = time.perf_counter()
    finally:
        await client.aclose()

    assert first.allowed and limiter.degraded
    assert [d.allowed for d in rest] == [True, True, False]  # still limited, locally
    # Only the first request pays for discovering the outage; the rest skip Redis.
    assert after_rest - after_first < 0.05
    assert after_first - start < 1.5
