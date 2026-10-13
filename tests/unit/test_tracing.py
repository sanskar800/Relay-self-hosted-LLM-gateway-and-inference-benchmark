"""OpenTelemetry spans: stages, attempts, attributes, log correlation, no prompts."""

import io
import json
import logging

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from relay.backends.base import BackendUnavailable
from relay.config import RelayConfig
from tests.fake_engine import FakeEngine
from tests.unit.fakes import FakeBackend, FakeStream, make_app, make_client, make_config

SECRET = "SECRET-PROMPT-TEXT-456"
BODY = {"model": "m", "messages": [{"role": "user", "content": SECRET}], "max_tokens": 5}


def tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def by_name(spans) -> dict:
    return {s.name: s for s in spans}


def test_stream_request_produces_one_trace_with_relay_stages() -> None:
    provider, exporter = tracing()
    app = make_app(
        config=make_config("m", "b1"),
        transport=FakeEngine(tokens=["a", "b", "c"]).transport(),
        tracer_provider=provider,
    )
    with make_client(app) as client:
        resp = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert resp.status_code == 200
    spans = exporter.get_finished_spans()
    names = by_name(spans)
    server = names["POST /v1/chat/completions"]
    for stage in ("relay.auth", "relay.rate_limit", "relay.route", "relay.backend_attempt"):
        assert stage in names, stage
    assert len({s.context.trace_id for s in spans}) == 1  # all one trace

    route, attempt = names["relay.route"], names["relay.backend_attempt"]
    assert attempt.parent.span_id == route.context.span_id
    http_calls = [s for s in spans if s.name == "POST" and s.parent]
    assert any(s.parent.span_id == attempt.context.span_id for s in http_calls)  # httpx

    attrs = server.attributes
    assert attrs["relay.tenant"] == "team-test" and attrs["gen_ai.request.model"] == "m"
    assert attrs["gen_ai.usage.output_tokens"] == 3 and attrs["relay.backend"] == "b1"
    assert attrs["relay.request_id"] == resp.headers["x-request-id"]
    assert "first_token" in [e.name for e in server.events]
    # One server span, not one span per streamed chunk.
    assert sum(s.name.startswith("POST /v1/chat/completions") for s in spans) == 1


def test_retries_and_fallback_are_separate_attempt_spans() -> None:
    provider, exporter = tracing()
    config = RelayConfig.model_validate(
        {
            "backends": {
                "primary": {"engine": "llamacpp", "url": "http://p"},
                "secondary": {"engine": "llamacpp", "url": "http://s"},
            },
            "models": {"m": {"backends": ["primary", "secondary"]}},
            "retry": {"base_delay_s": 0},
        }
    )
    backends = {
        "primary": FakeBackend("primary", error=BackendUnavailable("primary", "down")),
        "secondary": FakeBackend("secondary", stream=FakeStream([b"data: [DONE]\n\n"])),
    }
    app = make_app(config=config, backends=backends, tracer_provider=provider)
    with make_client(app) as client:
        client.post("/v1/chat/completions", json=BODY)
    attempts = [
        (
            s.attributes["relay.backend"],
            s.attributes["relay.attempt"],
            s.attributes["relay.outcome"],
        )
        for s in exporter.get_finished_spans()
        if s.name == "relay.backend_attempt"
    ]
    assert attempts == [
        ("primary", 1, "BackendUnavailable"),
        ("primary", 2, "BackendUnavailable"),
        ("primary", 3, "BackendUnavailable"),
        ("secondary", 1, "ok"),
    ]


def test_log_lines_carry_the_trace_id() -> None:
    provider, exporter = tracing()
    app = make_app(
        config=make_config("m", "b1"), transport=FakeEngine().transport(), tracer_provider=provider
    )
    relay_logger = logging.getLogger("relay")
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(relay_logger.handlers[0].formatter)
    relay_logger.addHandler(handler)
    try:
        with make_client(app) as client:
            client.post("/v1/chat/completions", json=BODY)
    finally:
        relay_logger.removeHandler(handler)
    access = [json.loads(x) for x in buffer.getvalue().splitlines() if '"request"' in x][0]
    server = by_name(exporter.get_finished_spans())["POST /v1/chat/completions"]
    assert access["trace_id"] == format(server.context.trace_id, "032x")


def test_no_prompt_text_in_any_span() -> None:
    provider, exporter = tracing()
    app = make_app(
        config=make_config("m", "b1"), transport=FakeEngine().transport(), tracer_provider=provider
    )
    with make_client(app) as client:
        client.post("/v1/chat/completions", json={**BODY, "stream": True})
    dumped = "".join(s.to_json() for s in exporter.get_finished_spans())
    assert SECRET not in dumped


def test_probes_are_not_traced() -> None:
    provider, exporter = tracing()
    app = make_app(config=make_config(), tracer_provider=provider)
    with make_client(app) as client:
        client.get("/healthz")
        client.get("/readyz")
        client.get("/metrics")
    assert exporter.get_finished_spans() == ()
