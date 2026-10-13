"""Request ids, JSON access logs (without prompts) and Prometheus metrics."""

import io
import json
import logging
import re
from contextlib import contextmanager

from relay.accounting.usage import UsageEvent
from tests.fake_engine import FakeEngine
from tests.unit.fakes import make_app, make_client, make_config

SECRET = "SECRET-PROMPT-TEXT-123"
BODY = {"model": "m", "messages": [{"role": "user", "content": SECRET}], "max_tokens": 5}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


@contextmanager
def captured_logs():
    """Capture Relay's log output, rendered by its own JSON formatter."""
    relay_logger = logging.getLogger("relay")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(relay_logger.handlers[0].formatter)
    relay_logger.addHandler(handler)
    try:
        yield buffer
    finally:
        relay_logger.removeHandler(handler)


def app_and_events(engine: FakeEngine | None = None):
    events: list[UsageEvent] = []
    app = make_app(
        config=make_config("m", "b1"),
        transport=(engine or FakeEngine(tokens=["a", "b", "c", "d"])).transport(),
        usage_sink=events.append,
    )
    return app, events


def test_every_response_has_a_request_id_even_before_auth() -> None:
    app, _ = app_and_events()
    with make_client(app) as client:
        ok = client.post("/v1/chat/completions", json=BODY)
        unauthorised = client.post(
            "/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer rk_nope"}
        )
        unknown_route = client.get("/v1/nope")
        health = client.get("/healthz")
    for resp in (ok, unauthorised, unknown_route, health):
        assert UUID.fullmatch(resp.headers["x-request-id"])
    assert unauthorised.status_code == 401


def test_request_id_matches_the_ledger_event() -> None:
    app, events = app_and_events()
    with make_client(app) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert str(events[0].request_id) == resp.headers["x-request-id"]


def test_access_log_is_json_with_request_fields_and_no_prompt() -> None:
    app, _ = app_and_events()
    with captured_logs() as logs, make_client(app) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    lines = [json.loads(line) for line in logs.getvalue().splitlines() if line.strip()]
    (access,) = [x for x in lines if x.get("event") == "request"]
    assert access["request_id"] == resp.headers["x-request-id"]
    assert access["status"] == 200 and access["path"] == "/v1/chat/completions"
    assert (access["tenant"], access["model"], access["backend"]) == ("team-test", "m", "b1")
    assert access["stream"] is True and access["completion_tokens"] == 4
    assert access["ttft_ms"] is not None and access["duration_ms"] >= access["ttft_ms"]
    assert access["level"] == "info" and "timestamp" in access
    assert SECRET not in logs.getvalue()  # prompts are never logged


def test_probes_are_not_logged() -> None:
    app, _ = app_and_events()
    with captured_logs() as logs, make_client(app) as client:
        client.get("/healthz")
        client.get("/readyz")
        client.get("/metrics")
    assert '"event": "request"' not in logs.getvalue()


def test_metrics_cover_latency_tokens_and_state_without_ids() -> None:
    app, _ = app_and_events(FakeEngine(tokens=["a", "b", "c", "d"], prompt_tokens=7))
    with make_client(app) as client:
        client.post("/v1/chat/completions", json={**BODY, "stream": True})
        client.post("/v1/chat/completions", json=BODY)
        client.get("/v1/models/m")
        text = client.get("/metrics").text

    def value(sample: str) -> float:
        match = re.search(rf"^{re.escape(sample)} (\S+)$", text, re.M)
        assert match, f"missing: {sample}"
        return float(match.group(1))

    assert value('relay_ttft_seconds_count{backend="b1",model="m"}') == 1  # the stream
    assert value('relay_inter_token_latency_seconds_count{backend="b1",model="m"}') == 3  # 4 tokens
    assert value('relay_completion_tokens_total{model="m",tenant="team-test"}') == 8  # 4 + 4
    assert value('relay_prompt_tokens_total{model="m",tenant="team-test"}') == 14
    assert value('relay_backend_requests_total{backend="b1",outcome="ok"}') == 2
    assert value("relay_gateway_overhead_seconds_count") == 2
    assert value('relay_circuit_breaker_state{backend="b1"}') == 0
    assert value("relay_in_flight_requests") == 0
    assert (
        value('relay_http_requests_total{method="POST",route="/v1/chat/completions",status="200"}')
        == 2
    )
    # Routes are templates, not raw paths (bounded label values).
    assert 'route="/v1/models/{model_id:path}"' in text
    assert not UUID.search(text)  # no request ids (or any ids) as label values
    assert SECRET not in text


def test_rate_limited_requests_are_counted_per_tenant_and_kind() -> None:
    from relay.limits.token_bucket import InMemoryRateLimiter
    from tests.unit.fakes import TEST_KEY, StaticKeyStore, make_principal

    app = make_app(
        config=make_config("m", "b1"),
        transport=FakeEngine().transport(),
        key_store=StaticKeyStore({TEST_KEY: make_principal(requests_per_minute=1)}),
        rate_limiter=InMemoryRateLimiter(),
    )
    with make_client(app) as client:
        client.post("/v1/chat/completions", json=BODY)
        assert client.post("/v1/chat/completions", json=BODY).status_code == 429
        text = client.get("/metrics").text
    assert 'relay_rate_limited_total{kind="requests",tenant="team-test"} 1.0' in text
