"""OpenTelemetry tracing: one trace per request, exported over OTLP (e.g. to Jaeger).

Off unless RELAY_OTEL_ENDPOINT is set (tests and CI need no collector). Automatic
instrumentation gives the server span (FastAPI) and one span per backend HTTP call
(httpx); Relay adds spans for its own stages (auth, rate limit, routing, attempts).
Bodies (prompts, completions) are never recorded.
"""

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import relay

NOT_TRACED = "healthz,readyz,metrics"  # probes and scrapes


def build_tracer_provider(endpoint: str | None) -> trace.TracerProvider:
    """An SDK provider exporting to `endpoint` (OTLP/HTTP), or a no-op provider."""
    if not endpoint:
        return trace.NoOpTracerProvider()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "relay", "service.version": relay.__version__})
    )
    # Spans are batched in memory and exported in the background; an unreachable
    # collector costs dropped spans, never slower requests.
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    return provider


def instrument_app(app: FastAPI, provider: trace.TracerProvider) -> None:
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls=NOT_TRACED,
        # By default every ASGI send/receive gets a span: one per streamed token.
        exclude_spans=["send", "receive"],
    )


def instrument_backend_client(client: object, provider: trace.TracerProvider) -> None:
    """A span (and a traceparent header) for each HTTP call to a backend."""
    HTTPXClientInstrumentor.instrument_client(client, tracer_provider=provider)


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None
