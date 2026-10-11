"""The usage_daily view against the real Postgres (inside a rolled-back transaction)."""

import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

pytestmark = pytest.mark.integration

TODAY = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)

INSERT = """
INSERT INTO usage_events (request_id, occurred_at, tenant_id, model, backend, stream,
    status_code, prompt_tokens, completion_tokens, usage_source, duration_ms, cost_gbp)
VALUES ($1, $2, 'it-report', $3, 'llamacpp-cpu', true, $4, $5, $6, $7, 100, $8)
"""


async def test_usage_daily_sums_and_flags_what_is_not_exact(pg: asyncpg.Connection) -> None:
    await pg.execute(
        "INSERT INTO tenants (id, name, requests_per_minute, tokens_per_minute)"
        " VALUES ('it-report', 'report test', 60, 1000)"
    )
    rows = [
        # when, model, status, prompt, completion, source, cost
        (TODAY, "small", 200, 30, 10, "engine", 0.001),
        (TODAY, "small", 200, 20, 5, "engine", None),  # unpriced
        (TODAY, "small", 200, None, 7, "estimated", None),  # e.g. client disconnected
        (TODAY, "small", 502, None, None, "missing", None),  # backend down
        (TODAY - timedelta(days=1), "small", 200, 50, 50, "engine", None),
        (TODAY, "large", 200, 100, 100, "engine", None),
    ]
    for when, model, status, prompt, completion, source, cost in rows:
        await pg.execute(
            INSERT, uuid.uuid4(), when, model, status, prompt, completion, source, cost
        )

    report = {
        (r["day"].isoformat(), r["model"]): dict(r)
        for r in await pg.fetch("SELECT * FROM usage_daily WHERE tenant_id = 'it-report'")
    }
    today_small = report[("2026-10-11", "small")]
    assert today_small["requests"] == 4
    assert today_small["failed_requests"] == 1
    assert (today_small["prompt_tokens"], today_small["completion_tokens"]) == (50, 22)
    assert today_small["total_tokens"] == 72
    assert today_small["requests_without_exact_usage"] == 2  # estimated + missing
    assert float(today_small["cost_gbp"]) == pytest.approx(0.001)  # partial sum...
    assert today_small["unpriced_requests"] == 3  # ...and flagged as such

    assert report[("2026-10-10", "small")]["total_tokens"] == 100  # separate day
    assert report[("2026-10-11", "large")]["cost_gbp"] is None  # no prices: NULL, not 0
