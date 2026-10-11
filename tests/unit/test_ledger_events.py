"""What each request hands to the usage ledger, and the ledger's non-blocking submit."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from relay.accounting.ledger import UsageLedger
from relay.accounting.usage import UsageEvent, UsageRecord
from relay.backends.base import BackendResponse, BackendUnavailable
from relay.config import ModelPrice, PricingConfig
from tests.fake_engine import FakeEngine
from tests.unit.fakes import (
    TEST_KEY,
    FakeBackend,
    StaticKeyStore,
    make_app,
    make_client,
    make_config,
    make_principal,
)

PRINCIPAL = make_principal(tenant_id="team-x", key_id="11111111-1111-1111-1111-111111111111")
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
PRICED = PricingConfig(
    basis="test rates",
    models={"m": ModelPrice(input_gbp_per_1m=1.0, output_gbp_per_1m=3.0)},
)


def run(*, backend=None, transport=None, pricing=None, stream=False):
    events: list[UsageEvent] = []
    kwargs = {"backends": {"b": backend}} if backend else {"transport": transport}
    app = make_app(
        config=make_config("m", "b"),
        key_store=StaticKeyStore({TEST_KEY: PRINCIPAL}),
        usage_sink=events.append,
        pricing=pricing or PricingConfig(basis="tests"),
        **kwargs,
    )
    with make_client(app) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": stream})
    return resp, events


def test_event_carries_request_id_tenant_key_and_timing() -> None:
    usage = b'{"usage": {"prompt_tokens": 9, "completion_tokens": 4}}'
    resp, events = run(backend=FakeBackend("b", response=BackendResponse(200, usage, "x")))
    (event,) = events
    assert str(event.request_id) == resp.headers["x-request-id"]
    assert (event.tenant_id, event.api_key_id) == ("team-x", PRINCIPAL.key_id)
    assert event.usage == UsageRecord("m", "b", False, 200, 9, 4, "engine")
    assert event.ttft_ms is None  # TTFT only means something for streams
    assert event.duration_ms > 0
    assert event.cost_gbp is None  # no rates known: no invented cost


def test_stream_event_has_ttft_within_duration() -> None:
    resp, events = run(transport=FakeEngine(tokens=["a", "b"]).transport(), stream=True)
    assert resp.status_code == 200
    (event,) = events
    assert event.usage.source == "engine" and event.usage.completion_tokens == 2
    assert event.ttft_ms is not None and 0 <= event.ttft_ms <= event.duration_ms


def test_cost_is_computed_when_rates_are_known() -> None:
    usage = b'{"usage": {"prompt_tokens": 1000, "completion_tokens": 2000}}'
    _, events = run(
        backend=FakeBackend("b", response=BackendResponse(200, usage, "x")), pricing=PRICED
    )
    assert events[0].cost_gbp == pytest.approx((1000 * 1.0 + 2000 * 3.0) / 1e6)


def test_failed_backend_call_is_recorded_without_tokens() -> None:
    resp, events = run(backend=FakeBackend("b", error=BackendUnavailable("b", "down")))
    assert resp.status_code == 502
    assert resp.headers["x-request-id"] == str(events[0].request_id)
    assert events[0].usage == UsageRecord("m", "b", False, 502, None, None, "missing")


def test_cost_is_unknown_when_any_input_is_unknown() -> None:
    assert PRICED.cost_gbp("m", None, 10) is None  # prompt not reported
    assert PRICED.cost_gbp("other", 10, 10) is None  # no price for this model
    assert PricingConfig(basis="x", models={"m": ModelPrice()}).cost_gbp("m", 1, 1) is None


def _event() -> UsageEvent:
    record = UsageRecord("m", "b", False, 200, 1, 1, "engine")
    return UsageEvent(uuid.uuid4(), datetime.now(UTC), "t", PRINCIPAL.key_id, record, None, 1, None)


async def test_submit_never_blocks_and_counts_drops_when_full() -> None:
    ledger = UsageLedger(pool=None, max_queue=3)  # writer not started: nothing drains it
    start = asyncio.get_running_loop().time()
    for _ in range(10):
        ledger.submit(_event())
    assert asyncio.get_running_loop().time() - start < 0.05
    assert (ledger.queued, ledger.dropped) == (3, 7)
