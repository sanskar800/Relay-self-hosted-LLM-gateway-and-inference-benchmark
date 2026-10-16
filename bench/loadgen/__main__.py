"""Run a benchmark scenario: `uv run python -m bench.loadgen bench/scenarios/<name>.yaml`."""

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

from dotenv import dotenv_values

from bench.loadgen.config import Scenario, load_scenario
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.loadgen")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"), help="results root")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="smoke run: save under <out>/_scratch (git-ignored), not as a result",
    )
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    # Unique per run: a re-run must not hit prompts an engine cached from the last run.
    nonce = secrets.token_hex(4)
    levels = asyncio.run(run(scenario, nonce))
    unit = "req/s" if scenario.load.mode == "open" else "workers (closed loop)"
    for _, summary in levels:
        print(f"== {summary.level:g} {unit}")
        print_summary(summary)
    root = args.out / "_scratch" if args.scratch else args.out
    results = [r for level_results, _ in levels for r in level_results]
    saved = write_run(root, scenario, results, [s for _, s in levels], prompt_salt=nonce)
    print(f"saved {len(results)} requests to {saved}")


if __name__ == "__main__":
    main()
