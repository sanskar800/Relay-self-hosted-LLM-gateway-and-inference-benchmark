"""Run a benchmark scenario: `uv run python -m bench.loadgen bench/scenarios/<name>.yaml`."""

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values

from bench.loadgen.config import Scenario, load_scenario
from bench.loadgen.memory import (
    MIN_AVAILABLE_DURING_RUN_GIB,
    MIN_AVAILABLE_TO_START_GIB,
    MemoryMonitor,
    available_gib,
)
from bench.loadgen.metrics import Summary, summarise
from bench.loadgen.results import write_run
from bench.loadgen.runner import (
    RequestResult,
    arrival_schedule,
    run_closed_loop,
    run_open_loop,
)
from bench.loadgen.workload import generate, stream


def api_key_for(scenario: Scenario) -> str | None:
    name = scenario.target.api_key_env
    if not name:
        return None
    key = os.environ.get(name) or dotenv_values(".env").get(name)
    if not key:
        sys.exit(f"scenario needs an API key in ${name} (environment or .env)")
    return key


async def run(scenario: Scenario, nonce: str) -> list[tuple[list[RequestResult], Summary]]:
    key = api_key_for(scenario)
    load = scenario.load
    out = []
    for n_workers in load.concurrency if load.mode == "closed" else []:
        print(f"-> {n_workers} closed-loop workers for {load.duration_s:.0f} s...", flush=True)
        # Same shape at every level, but new prompts: no cache hits on earlier levels.
        specs = stream(scenario.workload, salt=f"{nonce}:c{n_workers}")
        results = await run_closed_loop(scenario, n_workers, specs, api_key=key)
        out.append((results, summarise(results, load.warmup_s, scenario.slo)))
    for rate in load.arrival_rates_rps if load.mode == "open" else []:
        n = len(arrival_schedule(scenario, rate))
        specs = generate(scenario.workload, max(n, 1), salt=f"{nonce}:r{rate}")
        print(
            f"-> rate {rate} req/s for {scenario.load.duration_s:.0f} s ({n} requests)...",
            flush=True,
        )
        results = await run_open_loop(scenario, rate, specs, api_key=key)
        out.append((results, summarise(results, scenario.load.warmup_s, scenario.slo)))
    return out


def fmt(x: float | None, scale: float = 1000, unit: str = "ms") -> str:
    return "-" if x is None else f"{x * scale:.0f} {unit}" if unit == "ms" else f"{x:.2f}"


def print_summary(s: Summary) -> None:
    print(
        f"   sent {s.sent}  ok {s.ok}  errors {s.errors} ({s.error_rate:.0%})  "
        f"offered {s.offered_rps:.2f} req/s  throughput {s.throughput_rps:.2f} req/s  "
        f"{s.output_tokens_per_s:.1f} tok/s  "
        f"goodput {s.goodput_rps:.2f} req/s ({s.goodput_ratio:.0%})\n"
        f"   TTFT p50/p95/p99 {fmt(s.ttft_p50)} / {fmt(s.ttft_p95)} / {fmt(s.ttft_p99)}   "
        f"ITL p50/p95/p99 {fmt(s.itl_p50)} / {fmt(s.itl_p95)} / {fmt(s.itl_p99)}\n"
        f"   TPOT p50/p95 {fmt(s.tpot_p50)} / {fmt(s.tpot_p95)}   "
        f"E2E p50/p95 {fmt(s.e2e_p50)} / {fmt(s.e2e_p95)}   "
        f"scheduling lag p99 {fmt(s.scheduling_lag_p99)}"
    )


PROBE_LOAD = {"mode": "closed", "concurrency": [1, 2, 4, 8], "duration_s": 30, "warmup_s": 5}
PROBE_FRACTIONS = (0.25, 0.5, 0.75, 1.0, 1.25)


def probe_scenario(scenario: Scenario) -> Scenario:
    """A closed loop at rising concurrency: a quick estimate of where the configuration
    saturates, to choose the open-loop rates. Never a result (always saved to _scratch)."""
    raw = scenario.model_dump(mode="json")
    load = {**PROBE_LOAD, "request_timeout_s": scenario.load.request_timeout_s}
    return Scenario.model_validate({**raw, "name": f"{scenario.name}-probe", "load": load})


def suggested_rates(summaries: list[Summary]) -> tuple[float, list[float]]:
    capacity = max(s.throughput_rps for s in summaries)
    return capacity, [round(capacity * f, 2) for f in PROBE_FRACTIONS]


def unexpected_backends(scenario: Scenario, results: list[RequestResult]) -> set[str]:
    if scenario.setup is None:
        return set()
    return {r.backend for r in results if r.backend and r.backend != scenario.setup.backend}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.loadgen")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"), help="results root")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="smoke run: save under <out>/_scratch (git-ignored), not as a result",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="estimate saturation with a short closed loop and suggest open-loop rates",
    )
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help=f"start even with under {MIN_AVAILABLE_TO_START_GIB} GiB RAM available",
    )
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    if args.probe:
        scenario = probe_scenario(scenario)
    if (free := available_gib()) < MIN_AVAILABLE_TO_START_GIB and not args.allow_low_memory:
        sys.exit(
            f"only {free:.1f} GiB RAM available (need {MIN_AVAILABLE_TO_START_GIB}): a host "
            "that pages to disk measures itself, not the engine (docs/findings.md F8). "
            "Stop what the benchmark does not need, or pass --allow-low-memory."
        )
    # Unique per run: a re-run must not hit prompts an engine cached from the last run.
    nonce = secrets.token_hex(4)
    with MemoryMonitor() as memory:
        levels = asyncio.run(run(scenario, nonce))
    unit = "req/s" if scenario.load.mode == "open" else "workers (closed loop)"
    for _, summary in levels:
        print(f"== {summary.level:g} {unit}")
        print_summary(summary)
    root = args.out / "_scratch" if args.scratch or args.probe else args.out
    results = [r for level_results, _ in levels for r in level_results]
    saved = write_run(
        root,
        scenario,
        results,
        [s for _, s in levels],
        prompt_salt=nonce,
        extra_env={"host_memory": memory.report()},
    )
    print(f"saved {len(results)} requests to {saved}")
    m = memory.report()
    print(
        f"host RAM available: min {m['available_gib_min']} GiB (start {m['available_gib_start']})"
    )
    if args.probe:
        capacity, rates = suggested_rates([s for _, s in levels])
        fractions = ", ".join(f"x{f}" for f in PROBE_FRACTIONS)
        print(f"\nprobe: peak throughput {capacity:.2f} req/s (closed loop)")
        print(f"suggested arrival_rates_rps: {rates}  ({fractions} of peak)")
    if wrong := unexpected_backends(scenario, results):
        expected = scenario.setup.backend if scenario.setup else None
        sys.exit(
            f"INVALID RUN: requests served by {sorted(wrong)}, expected {expected}; "
            "do not use these results"
        )
    if memory.low:
        sys.exit(
            f"INVALID RUN: host RAM fell to {memory.report()['available_gib_min']} GiB "
            f"(threshold {MIN_AVAILABLE_DURING_RUN_GIB}); timings may reflect paging "
            "(docs/findings.md F8). Do not use these results"
        )


if __name__ == "__main__":
    main()
