"""The usage ledger against the real Postgres: counts, batching, outages, bad rows."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import asyncpg
import pytest

from relay.accounting.db import create_pool
from relay.accounting.ledger import UsageLedger
from relay.accounting.usage import UsageEvent, UsageRecord
from relay.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def pool():
    try:
        p = await create_pool(Settings().database_url)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not reachable: {exc!r}")
    yield p
    await p.close()


@pytest.fixture
async def tenant(pool):
    """A throwaway tenant; its usage rows are deleted afterwards (they are committed)."""
    tenant_id = f"it-ledger-{uuid.uuid4().hex[:8]}"
    await pool.execute(
        "INSERT INTO tenants (id, name, requests_per_minute, tokens_per_minute)"
        " VALUES ($1, 'ledger test', 60, 1000)",
        tenant_id,
    )
    yield tenant_id
    await pool.execute("DELETE FROM usage_events WHERE tenant_id = $1", tenant_id)
    await pool.execute("DELETE FROM tenants WHERE id = $1", tenant_id)


def event(tenant_id: str, request_id: uuid.UUID | None = None) -> UsageEvent:
    record = UsageRecord("qwen2.5-1.5b-instruct", "llamacpp-cpu", True, 200, 30, 5, "engine")
    return UsageEvent(
        request_id or uuid.uuid4(), datetime.now(UTC), tenant_id, None, record, 12.5, 300.0, None
    )


async def count(pool, tenant_id: str) -> int:
    return await pool.fetchval("SELECT count(*) FROM usage_events WHERE tenant_id = $1", tenant_id)


async def wait_for(predicate, timeout_s: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not await predicate():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.05)


async def test_n_events_become_n_rows_in_batches(pool, tenant) -> None:
    ledger = UsageLedger(pool, batch_size=50, flush_interval_s=0.2)
    ledger.start()
    for _ in range(120):
        ledger.submit(event(tenant))

    async def all_written() -> bool:
        return await count(pool, tenant) == 120

    await wait_for(all_written)
    await ledger.close()
    assert ledger.written == 120 and ledger.dropped == ledger.rejected == 0
    row = await pool.fetchrow(
        "SELECT prompt_tokens, completion_tokens, usage_source, ttft_ms, duration_ms"
        " FROM usage_events WHERE tenant_id = $1 LIMIT 1",
        tenant,
    )
    assert tuple(row) == (30, 5, "engine", 12.5, 300.0)


async def test_resubmitted_event_is_not_double_counted(pool, tenant) -> None:
    ledger = UsageLedger(pool)
    same = uuid.uuid4()
    await ledger._write([event(tenant, same)])
    await ledger._write([event(tenant, same)])  # e.g. a retry after a lost commit reply
    assert await count(pool, tenant) == 1


class FlakyPool:
    """Wraps the real pool; the first `failures` acquisitions fail like a dead database."""

    def __init__(self, pool, failures: int) -> None:
        self._pool, self.failures = pool, failures

    @asynccontextmanager
    async def acquire(self):
        if self.failures > 0:
            self.failures -= 1
            raise ConnectionRefusedError("database unavailable")
        async with self._pool.acquire() as conn:
            yield conn


async def test_postgres_outage_buffers_events_and_writes_them_later(pool, tenant) -> None:
    ledger = UsageLedger(
        FlakyPool(pool, failures=2),
        flush_interval_s=0.05,
        initial_backoff_s=0.05,
        max_backoff_s=0.1,
    )
    ledger.start()
    for _ in range(5):
        ledger.submit(event(tenant))  # requests are not slowed by the outage

    async def written() -> bool:
        return await count(pool, tenant) == 5

    await wait_for(written, timeout_s=15)
    await ledger.close()
    assert ledger.write_failures == 2 and ledger.dropped == 0


async def test_a_bad_row_is_rejected_without_losing_the_rest(pool, tenant) -> None:
    ledger = UsageLedger(pool)
    good = [event(tenant) for _ in range(3)]
    bad = event("no-such-tenant")  # violates the tenants foreign key
    await ledger._write([good[0], bad, good[1], good[2]])
    assert await count(pool, tenant) == 3
    assert ledger.rejected == 1 and ledger.written == 3


async def test_close_flushes_what_is_still_queued(pool, tenant) -> None:
    ledger = UsageLedger(pool)  # writer never started: everything is still queued
    for _ in range(7):
        ledger.submit(event(tenant))
    await ledger.close()
    assert await count(pool, tenant) == 7
