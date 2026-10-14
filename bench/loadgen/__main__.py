"""Run a benchmark scenario: `uv run python -m bench.loadgen bench/scenarios/<name>.yaml`."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from bench.loadgen.config import Scenario, load_scenario
from bench.loadgen.metrics import Summary, summarise
from bench.loadgen.runner import RequestResult, arrival_schedule, run_open_loop
from bench.loadgen.workload import generate


def api_key_for(scenario: Scenario) -> str | None:
    name = scenario.target.api_key_env
    if not name:
        return None
    key = os.environ.get(name) or dotenv_values(".env").get(name)
    if not key:
        sys.exit(f"scenario needs an API key in ${name} (environment or .env)")
    return key


async def run(scenario: Scenario) -> list[tuple[list[RequestResult], Summary]]:
    key = api_key_for(scenario)
    out = []
    for rate in scenario.load.arrival_rates_rps:
        n = len(arrival_schedule(scenario, rate))
        specs = generate(scenario.workload, max(n, 1))
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
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    if scenario.load.mode != "open":
        sys.exit("only open-loop scenarios are supported so far")
    for _, summary in asyncio.run(run(scenario)):
        print(f"== rate {summary.level} req/s")
        print_summary(summary)


if __name__ == "__main__":
    main()
