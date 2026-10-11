"""Integration tests run against the real Compose services (Postgres, Redis).

Start them first: `make up`. Locally the tests skip if a service is unreachable; CI sets
RELAY_REQUIRE_STACK=1 so a missing service fails instead of silently skipping.
"""

import os
from collections.abc import AsyncIterator

import asyncpg
import pytest
import redis.asyncio as aioredis

from relay.config import Settings

SETTINGS = Settings()


def _unavailable(what: str, exc: Exception) -> None:
    message = f"{what} not reachable ({exc!r}); run `make up`"
    if os.environ.get("RELAY_REQUIRE_STACK") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
async def pg() -> AsyncIterator[asyncpg.Connection]:
    """A connection inside a transaction that is always rolled back: tests leave no data."""
    try:
        conn = await asyncpg.connect(SETTINGS.database_url, timeout=3)
    except (OSError, asyncpg.PostgresError) as exc:
        _unavailable("Postgres", exc)
    tx = conn.transaction()
    await tx.start()
    try:
        yield conn
    finally:
        await tx.rollback()
        await conn.close()


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    client = aioredis.from_url(SETTINGS.redis_url, socket_connect_timeout=3)
    try:
        await client.ping()
    except (OSError, aioredis.RedisError) as exc:
        await client.aclose()
        _unavailable("Redis", exc)
    try:
        yield client
    finally:
        await client.aclose()
