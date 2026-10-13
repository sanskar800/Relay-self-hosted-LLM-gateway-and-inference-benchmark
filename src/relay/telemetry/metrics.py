"""Prometheus metrics, exposed at GET /metrics.

Latency is recorded in histograms (so p50/p95/p99 can be computed, also across
instances); counts and tokens in counters; current state in gauges. Labels have few,
fixed values (route, status, model, backend, tenant): never ids or text, because each
distinct label combination is a separate time series.
"""

from collections.abc import Callable, Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30)
ITL_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)
OVERHEAD_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25)
DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)
BREAKER_STATE = {"closed": 0, "half_open": 1, "open": 2}


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.http_requests = Counter(
            "relay_http_requests",
            "HTTP requests handled",
            ["route", "method", "status"],
            registry=r,
        )
        self.http_duration = Histogram(
            "relay_http_request_duration_seconds", "Time to complete the response, incl. streaming",
            ["route", "method"], buckets=DURATION_BUCKETS, registry=r,
        )  # fmt: skip
        self.ttft = Histogram(
            "relay_ttft_seconds", "Time from request start to the first content token",
            ["model", "backend"], buckets=TTFT_BUCKETS, registry=r,
        )  # fmt: skip
        self.itl = Histogram(
            "relay_inter_token_latency_seconds", "Time between consecutive content tokens",
            ["model", "backend"], buckets=ITL_BUCKETS, registry=r,
        )  # fmt: skip
        self.gateway_overhead = Histogram(
            "relay_gateway_overhead_seconds",
            "Relay's own time before calling a backend (auth, policy, limits, routing)",
            buckets=OVERHEAD_BUCKETS, registry=r,
        )  # fmt: skip
        self.prompt_tokens = Counter(
            "relay_prompt_tokens", "Prompt tokens", ["tenant", "model"], registry=r
        )
        self.completion_tokens = Counter(
            "relay_completion_tokens", "Completion tokens", ["tenant", "model"], registry=r
        )
        self.cost_gbp = Counter(
            "relay_cost_gbp", "Estimated cost in GBP (only priced requests)",
            ["tenant", "model"], registry=r,
        )  # fmt: skip
        self.backend_requests = Counter(
            "relay_backend_requests", "Chat requests by serving backend and outcome",
            ["backend", "outcome"], registry=r,
        )  # fmt: skip
        self.retries = Counter(
            "relay_retries", "Retries before the first token", ["backend"], registry=r
        )
        self.fallbacks = Counter(
            "relay_fallbacks", "Requests moved to the next backend", ["model", "from_backend"],
            registry=r,
        )  # fmt: skip
        self.backend_in_flight = Gauge(
            "relay_backend_in_flight", "Open streams per backend", ["backend"], registry=r
        )
        self.rate_limited = Counter(
            "relay_rate_limited", "Requests rejected by rate limits", ["tenant", "kind"],
            registry=r,
        )  # fmt: skip

    def watch(self, collector: Collector) -> None:
        self.registry.register(collector)


class StateCollector(Collector):
    """Reads live state at scrape time (breakers, in-flight, limiter, ledger)."""

    def __init__(
        self,
        breakers: Callable[[], dict[str, str]],
        in_flight: Callable[[], int],
        limiter_degraded: Callable[[], bool | None],
        ledger: Callable[[], dict[str, int] | None],
    ) -> None:
        self._breakers = breakers
        self._in_flight = in_flight
        self._limiter_degraded = limiter_degraded
        self._ledger = ledger

    def collect(self) -> Iterable:
        state = GaugeMetricFamily(
            "relay_circuit_breaker_state",
            "Circuit breaker per backend: 0 closed, 1 half-open, 2 open",
            labels=["backend"],
        )
        for backend, value in self._breakers().items():
            state.add_metric([backend], BREAKER_STATE[value])
        yield state
        yield GaugeMetricFamily(
            "relay_in_flight_requests", "API requests being served", value=self._in_flight()
        )
        degraded = self._limiter_degraded()
        if degraded is not None:
            yield GaugeMetricFamily(
                "relay_ratelimiter_degraded",
                "1 while Redis is unreachable and limits are enforced per instance (ADR-002)",
                value=int(degraded),
            )
        ledger = self._ledger()
        if ledger is not None:
            yield GaugeMetricFamily(
                "relay_ledger_queue_depth", "Usage events waiting to be written",
                value=ledger["queued"],
            )  # fmt: skip
            events = CounterMetricFamily(
                "relay_ledger_events", "Usage events by outcome", labels=["outcome"]
            )
            for outcome in ("written", "dropped", "rejected"):
                events.add_metric([outcome], ledger[outcome])
            yield events
