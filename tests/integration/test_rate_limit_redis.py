"""The Lua token bucket against the real Redis: limits, atomicity, expiry, settling."""

import asyncio
import uuid

import pytest
import redis.asyncio as aioredis

from relay.limits.token_bucket import RedisRateLimiter
from tests.unit.fakes import make_principal

pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant(redis_client: aioredis.Redis):
    tenant_id = f"it-{uuid.uuid4().hex[:8]}"
    yield tenant_id
    await redis_client.delete(f"relay:rl:{{{tenant_id}}}:req", f"relay:rl:{{{tenant_id}}}:tok")


async def test_n_plus_one_rapid_requests_last_is_denied(redis_client, tenant) -> None:
    limiter = RedisRateLimiter(redis_client)
    p = make_principal(tenant_id=tenant, requests_per_minute=5, tokens_per_minute=10_000)
    results = [await limiter.acquire(p, 10) for _ in range(6)]
    assert [r.allowed for r in results] == [True] * 5 + [False]
    assert results[-1].retry_after_s == pytest.approx(12, abs=0.5)  # 1 / (5 per 60 s)


async def test_concurrent_requests_cannot_overspend_the_bucket(redis_client, tenant) -> None:
    limiter = RedisRateLimiter(redis_client)
    p = make_principal(tenant_id=tenant, requests_per_minute=10, tokens_per_minute=10_000)
    results = await asyncio.gather(*(limiter.acquire(p, 1) for _ in range(50)))
    assert sum(r.allowed for r in results) == 10  # exactly the capacity, never more


async def test_control_naive_read_then_write_does_overspend(redis_client, tenant) -> None:
    """Why the Lua script exists: GET, check in Python, SET is not atomic."""
    key = f"relay:test:naive:{tenant}"
    await redis_client.set(key, 10)

    async def naive_acquire() -> bool:
        tokens = int(await redis_client.get(key))
        if tokens < 1:
            return False
        await redis_client.set(key, tokens - 1)
        return True

    admitted = sum(await asyncio.gather(*(naive_acquire() for _ in range(50))))
    await redis_client.delete(key)
    assert admitted > 10  # the race lets many more requests through than the limit


async def test_keys_expire_when_idle(redis_client, tenant) -> None:
    limiter = RedisRateLimiter(redis_client)
    await limiter.acquire(make_principal(tenant_id=tenant), 1)
    ttl = await redis_client.ttl(f"relay:rl:{{{tenant}}}:tok")
    assert 60 < ttl <= 120  # full refill time (60 s) + 60 s


async def test_settle_refunds_in_redis(redis_client, tenant) -> None:
    limiter = RedisRateLimiter(redis_client)
    p = make_principal(tenant_id=tenant, requests_per_minute=100, tokens_per_minute=100)
    assert (await limiter.acquire(p, 90)).allowed
    assert not (await limiter.acquire(p, 50)).allowed
    await limiter.settle(p, 80)  # reserved 90, actually used 10
    assert (await limiter.acquire(p, 50)).allowed
