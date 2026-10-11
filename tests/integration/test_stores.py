"""The Compose Postgres and Redis are reachable and the schema enforces its rules."""

import hashlib
import uuid
from datetime import UTC, datetime

import asyncpg
import pytest
import redis.asyncio as aioredis

from relay.accounting.db import create_pool
from relay.config import Settings

pytestmark = pytest.mark.integration


async def add_tenant(pg: asyncpg.Connection, tenant: str = "team-a") -> None:
    await pg.execute(
        "INSERT INTO tenants (id, name, allowed_models, requests_per_minute, tokens_per_minute)"
        " VALUES ($1, 'Team A', '{qwen2.5-1.5b-instruct}', 60, 20000)",
        tenant,
    )


async def test_schema_tables_exist(pg: asyncpg.Connection) -> None:
    rows = await pg.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    assert {r["table_name"] for r in rows} >= {"tenants", "api_keys", "usage_events"}


async def test_tenant_and_hashed_key_round_trip(pg: asyncpg.Connection) -> None:
    await add_tenant(pg)
    key_hash = hashlib.sha256(b"rk_example").digest()
    await pg.execute(
        "INSERT INTO api_keys (tenant_id, key_hash, key_prefix) VALUES ('team-a', $1, 'rk_exam')",
        key_hash,
    )
    row = await pg.fetchrow(
        "SELECT t.id, t.allowed_models FROM api_keys k JOIN tenants t ON t.id = k.tenant_id"
        " WHERE k.key_hash = $1 AND k.revoked_at IS NULL",
        key_hash,
    )
    assert row["id"] == "team-a"
    assert row["allowed_models"] == ["qwen2.5-1.5b-instruct"]


@pytest.mark.parametrize(
    ("sql", "constraint"),
    [
        (
            "INSERT INTO tenants (id, name, requests_per_minute, tokens_per_minute)"
            " VALUES ('Bad Id!', 'x', 1, 1)",
            "tenants_id_check",
        ),
        (
            "INSERT INTO tenants (id, name, requests_per_minute, tokens_per_minute)"
            " VALUES ('team-b', 'x', 0, 1)",
            "tenants_requests_per_minute_check",
        ),
        (
            "INSERT INTO api_keys (tenant_id, key_hash, key_prefix)"
            " VALUES ('team-a', '\\x00'::bytea, 'rk_x')",
            "api_keys_key_hash_check",
        ),
    ],
)
async def test_schema_rejects_invalid_rows(
    pg: asyncpg.Connection, sql: str, constraint: str
) -> None:
    await add_tenant(pg)
    with pytest.raises(asyncpg.CheckViolationError) as excinfo:
        await pg.execute(sql)
    assert excinfo.value.constraint_name == constraint


async def test_usage_event_request_id_is_idempotent(pg: asyncpg.Connection) -> None:
    await add_tenant(pg)
    event = (
        uuid.uuid4(),
        datetime.now(UTC),
        "team-a",
        "qwen2.5-1.5b-instruct",
        "llamacpp-cpu",
        True,
        200,
        35,
        24,
        "engine",
        120.5,
    )
    insert = (
        "INSERT INTO usage_events (request_id, occurred_at, tenant_id, model, backend, stream,"
        " status_code, prompt_tokens, completion_tokens, usage_source, duration_ms)"
        " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
        " ON CONFLICT (request_id) DO NOTHING"
    )
    await pg.execute(insert, *event)
    await pg.execute(insert, *event)  # a retried batch must not double-count
    assert await pg.fetchval("SELECT count(*) FROM usage_events") == 1


async def test_pool_connects_with_configured_url() -> None:
    try:
        pool = await create_pool(Settings().database_url)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    try:
        assert await pool.fetchval("SELECT 1") == 1
    finally:
        await pool.close()


async def test_redis_round_trip_with_expiry(redis_client: aioredis.Redis) -> None:
    key = f"relay:test:{uuid.uuid4()}"
    await redis_client.set(key, "1", ex=5)
    assert await redis_client.get(key) == b"1"
    assert 0 < await redis_client.ttl(key) <= 5
    await redis_client.delete(key)
