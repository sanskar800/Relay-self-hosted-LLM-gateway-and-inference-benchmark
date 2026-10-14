"""Metric definitions on hand-built timelines with known answers, and the open loop."""

import asyncio
import statistics
import time

import pytest
from bench.loadgen.config import Scenario, Slo
from bench.loadgen.metrics import itl, percentile, summarise, tpot, ttft
from bench.loadgen.runner import RequestResult, poisson_schedule, run_open_loop
from bench.loadgen.workload import generate

from tests.fake_engine import FakeEngine


def result(**kw) -> RequestResult:
    base = dict(level=1.0, index=0, scheduled_s=10.0, sent_s=10.0, status=200)
    return RequestResult(**{**base, **kw})


# --- per-request definitions ----------------------------------------------------------


def test_definitions_on_a_known_timeline() -> None:
    # Sent at 10.0, first token at 10.5, then tokens every 0.1 s, 5 tokens, end at 10.9.
    r = result(
        first_token_s=10.5,
        token_times_s=[10.5, 10.6, 10.7, 10.8, 10.9],
        end_s=10.9,
        completion_tokens=5,
    )
    assert ttft(r) == pytest.approx(0.5)
    assert itl(r) == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert tpot(r) == pytest.approx(0.1)  # (10.9 - 10.5) / (5 - 1)


def test_tpot_uses_the_engines_token_count_not_events() -> None:
    # Two events carrying 9 tokens (batched under load): ITL sees 1 gap, TPOT 8 tokens.
    r = result(first_token_s=1.0, token_times_s=[1.0, 1.8], end_s=1.8, completion_tokens=9)
    assert itl(r) == pytest.approx([0.8])
    assert tpot(r) == pytest.approx(0.1)


def test_percentile_matches_linear_interpolation() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile(values, 0) == 1 and percentile(values, 100) == 10
    assert percentile([], 50) is None


# --- aggregates -----------------------------------------------------------------------


def test_summary_excludes_warmup_and_computes_rates_and_goodput() -> None:
    slo = Slo(ttft_s=1.0, tpot_s=0.1)
    fast = dict(token_times_s=[], completion_tokens=11)
    results = [
        # warm-up request: ignored
        result(index=0, scheduled_s=1.0, sent_s=1.0, first_token_s=9.0, end_s=9.5, in_warmup=True,
               completion_tokens=2),
        # meets SLO: TTFT 0.5, TPOT (12.5-10.5)/10 = 0.2 -> NO, TPOT too slow
        result(index=1, sent_s=10.0, first_token_s=10.5, end_s=12.5, **fast),
        # meets SLO: TTFT 0.2, TPOT 0.05
        result(index=2, sent_s=11.0, first_token_s=11.2, end_s=11.7, **fast),
        # misses: TTFT 2.0
        result(index=3, sent_s=12.0, first_token_s=14.0, end_s=14.5, **fast),
        # failed request
        result(index=4, sent_s=13.0, end_s=13.1, status=502, error="Backend unavailable"),
    ]  # fmt: skip
    s = summarise(results, warmup_s=5.0, slo=slo)
    assert (s.sent, s.ok, s.errors) == (4, 3, 1)
    assert s.error_rate == pytest.approx(0.25)
    assert s.window_s == pytest.approx(14.5 - 5.0)
    assert s.throughput_rps == pytest.approx(3 / 9.5)
    assert s.output_tokens_per_s == pytest.approx(33 / 9.5)
    assert s.goodput_ratio == pytest.approx(1 / 4)  # only request 2 meets both targets
    assert s.ttft_p50 == pytest.approx(0.5)


# --- the open loop --------------------------------------------------------------------


def test_poisson_schedule_has_the_configured_mean_rate() -> None:
    times = poisson_schedule(rate_rps=20, duration_s=500, seed=1)
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert statistics.mean(gaps) == pytest.approx(1 / 20, rel=0.05)
    # Exponential gaps: the standard deviation equals the mean (bursts happen).
    assert statistics.stdev(gaps) == pytest.approx(1 / 20, rel=0.1)
    assert poisson_schedule(20, 10, seed=1) == poisson_schedule(20, 10, seed=1)


SCENARIO = Scenario.model_validate(
    {
        "name": "t",
        "target": {"base_url": "http://engine/v1", "model": "m"},
        "load": {"arrival_rates_rps": [40], "duration_s": 1.0, "warmup_s": 0},
        "workload": {
            "prompt_tokens": {"median": 20, "sigma": 0, "min": 1, "max": 100},
            "output_tokens": {"median": 4, "sigma": 0, "min": 1, "max": 100},
        },
    }
)


def test_arrivals_are_not_slowed_by_a_slow_server() -> None:
    # Each request takes 0.5 s to its first token, but arrivals are every ~25 ms: an open
    # loop keeps sending on schedule (a closed loop would wait).
    engine = FakeEngine(tokens=["a", "b"], first_event_delay_s=0.5)
    specs = generate(SCENARIO.workload, 200)
    start = time.perf_counter()
    results = asyncio.run(run_open_loop(SCENARIO, 40, specs, transport=engine.transport()))
    elapsed = time.perf_counter() - start
    lags = [r.sent_s - r.scheduled_s for r in results]
    assert len(results) > 20
    assert max(lags) < 0.05  # every request sent on time despite the slow server
    assert all(r.ok for r in results)
    assert elapsed < 1.0 + 0.5 + 0.5  # last arrival + one response, not N x 0.5 s
    assert all(ttft(r) == pytest.approx(0.5, abs=0.1) for r in results)


def test_errors_and_usage_are_recorded() -> None:
    engine = FakeEngine(tokens=["a", "b", "c"], prompt_tokens=9)
    specs = generate(SCENARIO.workload, 200)
    results = asyncio.run(run_open_loop(SCENARIO, 40, specs, transport=engine.transport()))
    assert all((r.prompt_tokens, r.completion_tokens) == (9, 3) for r in results)
    assert all(len(r.token_times_s) == 3 for r in results)
    sent = engine.requests[0]
    assert sent["stream"] is True and sent["ignore_eos"] is True
