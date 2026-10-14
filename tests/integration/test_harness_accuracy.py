"""The harness measured against an engine whose real timings are known, over a real socket.

The fake engine records when it actually started answering and when each token left
(time.perf_counter, same process, same clock). Comparing request by request:
    harness TTFT - engine TTFT   = time the harness adds (network, HTTP, SSE parsing)
    harness gap  - engine gap    = error in each inter-token latency
The engine's scripted delays are NOT the ground truth: sleeps overshoot by several ms on
Windows, so the engine is slower than its settings. Its recorded timeline is the truth.
"""

import asyncio
from pathlib import Path

import pytest
import yaml
from bench.loadgen.config import Scenario
from bench.loadgen.metrics import percentile, ttft
from bench.loadgen.runner import RequestResult, arrival_schedule, run_open_loop
from bench.loadgen.workload import generate

from tests.engine_server import serve_engine
from tests.fake_engine import FakeEngine

# Real sockets and wall-clock timing: kept out of the fast unit run.
pytestmark = pytest.mark.integration

TOKENS = 10
# Accepted measurement error, set from 6 repeated runs on the dev laptop with headroom.
# Observed: added TTFT p50 2.4-5.5 ms, but ~1 in 100 requests hit a ~40 ms hiccup (thread
# scheduling on Windows; not new connections, checked), so slow requests are checked as a
# SHARE, not a tail percentile of a small sample. Token-gap error p95 0.5-2.4 ms, with rare
# single ~10 ms hiccups (so gaps are checked at p95 plus a max).
# Well below what we benchmark (TTFT in hundreds of ms, token gaps in tens of ms). The
# engine runs in a thread of this process, sharing the GIL, so this is a worst case.
MAX_ADDED_TTFT_P50_S = 0.010
SLOW_ADDED_TTFT_S = 0.025
MAX_SLOW_SHARE = 0.10
MAX_GAP_ERROR_P95_S = 0.005
MAX_GAP_ERROR_S = 0.020
# The first request opens the connection (+40-60 ms observed): warm-up excludes it, as in
# every real run.
WARMUP_S = 0.5


def scenario_for(base_url: str, rate: float, duration_s: float) -> Scenario:
    raw = yaml.safe_load(Path("bench/scenarios/local-smoke.yaml").read_text(encoding="utf-8"))
    raw["target"] = {"base_url": f"{base_url}/v1", "model": "fake-model"}
    raw["load"].update(arrival_rates_rps=[rate], duration_s=duration_s, warmup_s=WARMUP_S)
    return Scenario.model_validate(raw)


def measure(rate: float, duration_s: float) -> list[tuple[RequestResult, dict]]:
    """Run the open loop against the fake engine; pair each result with the engine's
    own timeline for the same request (matched by the unique prompt)."""
    engine = FakeEngine(tokens=["tok"] * TOKENS, first_event_delay_s=0.2, token_delay_s=0.04)
    with serve_engine(engine) as url:
        scenario = scenario_for(url, rate, duration_s)
        specs = generate(scenario.workload, len(arrival_schedule(scenario, rate)))
        results = asyncio.run(run_open_loop(scenario, rate, specs))
    by_prompt = {t["prompt"]: t for t in engine.timelines}
    return [(r, by_prompt[specs[r.index].prompt]) for r in results if not r.in_warmup]


def errors(pairs: list[tuple[RequestResult, dict]]) -> tuple[list[float], list[float]]:
    added_ttft, gap_error = [], []
    for r, t in pairs:
        engine_ttft = t["tokens"][0] - t["received"]
        added_ttft.append(ttft(r) - engine_ttft)
        seen = r.token_times_s
        for i in range(1, TOKENS):
            harness_gap = seen[i] - seen[i - 1]
            engine_gap = t["tokens"][i] - t["tokens"][i - 1]
            gap_error.append(abs(harness_gap - engine_gap))
    return added_ttft, gap_error


@pytest.mark.parametrize(
    ("rate", "duration_s"),
    [
        (4.0, 8.0),  # light load: requests rarely overlap
        # ~10 requests in flight: if the harness itself became a bottleneck (event loop,
        # connection pool), its added time would grow here.
        (20.0, 4.0),
    ],
)
def test_harness_measures_what_the_engine_did(rate: float, duration_s: float) -> None:
    pairs = measure(rate, duration_s)
    assert len(pairs) >= 10
    assert all(r.ok and len(r.token_times_s) == TOKENS for r, _ in pairs)
    assert all(r.completion_tokens == TOKENS for r, _ in pairs)  # engine's count recorded

    added_ttft, gap_error = errors(pairs)
    # Causality: the harness never sees a token before the engine sent it.
    assert min(added_ttft) > 0
    assert percentile(added_ttft, 50) < MAX_ADDED_TTFT_P50_S
    slow = sum(a > SLOW_ADDED_TTFT_S for a in added_ttft)
    assert slow / len(added_ttft) <= MAX_SLOW_SHARE
    assert percentile(gap_error, 95) < MAX_GAP_ERROR_P95_S
    assert max(gap_error) < MAX_GAP_ERROR_S
