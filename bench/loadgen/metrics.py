"""Benchmark metrics from per-request timelines.

Definitions (per request, times in seconds):
- TTFT  = first content event - sent
- ITL   = gaps between consecutive content events (what a client experiences; an event
          may carry several tokens under load)
- TPOT  = (end - first content event) / (completion_tokens - 1), using the engine's count
- E2E   = end - sent
Aggregates exclude warm-up requests. Rates use the measurement window: from the end of the
warm-up to the last included request's end. Goodput counts successful requests meeting
BOTH SLO targets (TTFT <= slo.ttft_s and TPOT <= slo.tpot_s).
Percentiles use linear interpolation between closest ranks (as numpy's default).
"""

import math
from dataclasses import dataclass

from bench.loadgen.config import Slo
from bench.loadgen.runner import RequestResult


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def ttft(r: RequestResult) -> float | None:
    if r.first_token_s is None or r.sent_s is None:
        return None
    return r.first_token_s - r.sent_s


def tpot(r: RequestResult) -> float | None:
    if r.first_token_s is None or r.end_s is None or not r.completion_tokens:
        return None
    if r.completion_tokens < 2:
        return None
    return (r.end_s - r.first_token_s) / (r.completion_tokens - 1)


def itl(r: RequestResult) -> list[float]:
    t = r.token_times_s
    return [b - a for a, b in zip(t, t[1:], strict=False)]


def meets_slo(r: RequestResult, slo: Slo) -> bool:
    first, per_token = ttft(r), tpot(r)
    return (
        r.ok
        and first is not None
        and first <= slo.ttft_s
        and (per_token is None or per_token <= slo.tpot_s)
    )


@dataclass(frozen=True)
class Summary:
    level: float
    sent: int  # measured (non-warm-up) requests
    ok: int
    errors: int
    error_rate: float
    window_s: float
    offered_rps: float  # requests sent per second (should match the configured rate)
    throughput_rps: float  # successful requests completed per second
    output_tokens_per_s: float
    goodput_rps: float
    goodput_ratio: float  # share of sent requests that met the SLO
    ttft_p50: float | None
    ttft_p95: float | None
    ttft_p99: float | None
    itl_p50: float | None
    itl_p95: float | None
    itl_p99: float | None
    tpot_p50: float | None
    tpot_p95: float | None
    tpot_p99: float | None
    e2e_p50: float | None
    e2e_p95: float | None
    e2e_p99: float | None
    scheduling_lag_p99: float | None  # harness health: should be ~0


def summarise(results: list[RequestResult], warmup_s: float, slo: Slo) -> Summary:
    measured = [r for r in results if not r.in_warmup]
    ok = [r for r in measured if r.ok]
    ends = [r.end_s for r in measured if r.end_s is not None]
    window = max(ends) - warmup_s if ends else 0.0
    rate = (lambda n: n / window) if window > 0 else (lambda n: 0.0)
    ttfts = [x for r in ok if (x := ttft(r)) is not None]
    tpots = [x for r in ok if (x := tpot(r)) is not None]
    e2es = [r.end_s - r.sent_s for r in ok if r.end_s is not None and r.sent_s is not None]
    gaps = [g for r in ok for g in itl(r)]
    lags = [r.sent_s - r.scheduled_s for r in measured if r.sent_s is not None]
    good = [r for r in measured if meets_slo(r, slo)]
    n = len(measured)
    return Summary(
        level=results[0].level if results else 0.0,
        sent=n,
        ok=len(ok),
        errors=n - len(ok),
        error_rate=(n - len(ok)) / n if n else 0.0,
        window_s=window,
        offered_rps=rate(n),
        throughput_rps=rate(len(ok)),
        output_tokens_per_s=rate(sum(r.completion_tokens or 0 for r in ok)),
        goodput_rps=rate(len(good)),
        goodput_ratio=len(good) / n if n else 0.0,
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_p99=percentile(ttfts, 99),
        itl_p50=percentile(gaps, 50),
        itl_p95=percentile(gaps, 95),
        itl_p99=percentile(gaps, 99),
        tpot_p50=percentile(tpots, 50),
        tpot_p95=percentile(tpots, 95),
        tpot_p99=percentile(tpots, 99),
        e2e_p50=percentile(e2es, 50),
        e2e_p95=percentile(e2es, 95),
        e2e_p99=percentile(e2es, 99),
        scheduling_lag_p99=percentile(lags, 99),
    )
