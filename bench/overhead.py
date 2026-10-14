"""Gateway overhead: `uv run python -m bench.overhead [--scratch]`.

Serves a fake engine with FIXED latency, starts a real Relay (auth, Redis limiter,
Postgres ledger, metrics: the local Compose stack must be up) on port 8010 pointed at
it, and sends the same open-loop requests (same seed) twice:

    A  harness -> fake engine                 (direct)
    B  harness -> Relay -> same fake engine   (via Relay)

The engine's latency does not vary, so B - A is Relay's cost. A real engine's own
run-to-run variation would be larger than the overhead and hide it. Also reads Relay's
server-side `relay_gateway_overhead_seconds` (time before the upstream call) as a
cross-check. Both runs are saved as normal results folders.
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import yaml
from prometheus_client.parser import text_string_to_metric_families
from tests.engine_server import serve_engine  # the scripted engine the tests use
from tests.fake_engine import FakeEngine

from bench.loadgen.__main__ import api_key_for
from bench.loadgen.config import Scenario, load_scenario
from bench.loadgen.metrics import Summary, percentile, summarise, ttft
from bench.loadgen.results import write_run
from bench.loadgen.runner import RequestResult, arrival_schedule, run_open_loop
from bench.loadgen.workload import RequestSpec, generate

ENGINE = {"tokens": 32, "first_event_delay_s": 0.05, "token_delay_s": 0.02}
OVERHEAD_METRIC = "relay_gateway_overhead_seconds"


def relay_config(engine_url: str, model: str) -> dict:
    return {
        "backends": {"fake-engine": {"engine": "llamacpp", "url": engine_url}},
        "models": {model: {"backends": ["fake-engine"]}},
    }


def start_relay(config_path: Path, port: int, log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "RELAY_CONFIG": str(config_path), "RELAY_PORT": str(port)}
    # Own process group on Windows so Ctrl+Break reaches only Relay (graceful drain).
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    # Logs go to a file, not a pipe: an unread pipe fills up and blocks Relay mid-run.
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "relay.serve"],
            env=env, creationflags=flags, stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"Relay exited at start-up:\n{log_path.read_text(errors='replace')[-2000:]}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    stop_relay(proc)
    sys.exit("Relay did not become ready in 30 s (is `make up` running?)")


def stop_relay(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGTERM)
    try:
        proc.wait(15)
    except subprocess.TimeoutExpired:
        proc.kill()


def overhead_histogram(port: int) -> tuple[float, float]:
    """(sum, count) of Relay's server-side overhead histogram."""
    text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
    values = {}
    for family in text_string_to_metric_families(text):
        if family.name == OVERHEAD_METRIC:
            for sample in family.samples:
                values[sample.name] = sample.value
    return values.get(f"{OVERHEAD_METRIC}_sum", 0.0), values.get(f"{OVERHEAD_METRIC}_count", 0.0)


def requests_for(scenario: Scenario) -> list[RequestSpec]:
    rate = scenario.load.arrival_rates_rps[0]
    return generate(scenario.workload, len(arrival_schedule(scenario, rate)))


async def run_level(
    scenario: Scenario, specs: list[RequestSpec], api_key: str | None
) -> tuple[list[RequestResult], Summary]:
    rate = scenario.load.arrival_rates_rps[0]
    results = await run_open_loop(scenario, rate, specs, api_key=api_key)
    return results, summarise(results, scenario.load.warmup_s, scenario.slo)


def outside_engine(
    results: list[RequestResult], specs: list[RequestSpec], timelines: list[dict]
) -> dict[str, list[float]]:
    """Per request, time spent OUTSIDE the engine = what the harness saw minus what the
    engine actually did (its recorded timeline). This cancels the engine's own jitter
    (on Windows its sleeps snap to a ~15.6 ms timer), which would otherwise hide a
    few-ms overhead. Requests are matched by their unique prompt."""
    by_prompt = {t["prompt"]: t for t in timelines}
    ttft_added, stream_added = [], []
    for r in results:
        if r.in_warmup or not r.ok or not r.token_times_s:
            continue
        t = by_prompt[specs[r.index].prompt]
        ttft_added.append(ttft(r) - (t["tokens"][0] - t["received"]))
        seen = r.token_times_s[-1] - r.token_times_s[0]
        stream_added.append(seen - (t["tokens"][-1] - t["tokens"][0]))
    return {"ttft_added_s": ttft_added, "stream_added_s": stream_added}


def stats(values: list[float]) -> dict[str, float | None]:
    return {f"p{q}": percentile(values, q) for q in (50, 95, 99)}


def ms(x: float | None) -> str:
    return "-" if x is None else f"{x * 1000:.2f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m bench.overhead")
    parser.add_argument(
        "scenario", type=Path, nargs="?", default=Path("bench/scenarios/overhead.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--scratch", action="store_true", help="save under <out>/_scratch")
    args = parser.parse_args(argv)

    base = load_scenario(args.scenario)
    port = base.target.base_url.port
    key = api_key_for(base)
    engine = FakeEngine(
        tokens=["tok"] * ENGINE["tokens"],
        first_event_delay_s=ENGINE["first_event_delay_s"],
        token_delay_s=ENGINE["token_delay_s"],
    )
    runs: dict[str, tuple[Scenario, list[RequestResult], Summary, list[dict]]] = {}
    with serve_engine(engine) as engine_url, tempfile.TemporaryDirectory() as tmp:
        raw = base.model_dump(mode="json")
        via_relay = Scenario.model_validate({**raw, "name": f"{base.name}-relay"})
        direct = Scenario.model_validate(
            {
                **raw,
                "name": f"{base.name}-direct",
                "target": {**raw["target"], "base_url": f"{engine_url}/v1", "api_key_env": None},
            }
        )
        config_path = Path(tmp) / "relay.yaml"
        config_path.write_text(yaml.safe_dump(relay_config(engine_url, base.target.model)))

        print(f"fake engine {engine_url}: {ENGINE}", flush=True)
        print("-> direct to the engine...", flush=True)
        specs = requests_for(base)  # identical for both runs (same seed and load)
        results, summary = asyncio.run(run_level(direct, specs, None))
        runs["direct"] = (direct, results, summary, list(engine.timelines))
        engine.timelines.clear()  # same seed = same prompts: keep the two runs apart

        relay = start_relay(config_path, port, Path(tmp) / "relay.log")
        try:
            before = overhead_histogram(port)
            print(f"-> via Relay on :{port}...", flush=True)
            results, summary = asyncio.run(run_level(via_relay, specs, key))
            runs["relay"] = (via_relay, results, summary, list(engine.timelines))
            after = overhead_histogram(port)
        finally:
            stop_relay(relay)

    measured = {name: outside_engine(rs, specs, tl) for name, (_, rs, _, tl) in runs.items()}
    report: dict = {"engine": ENGINE, "requests": {}, "outside_engine": {}, "relay_adds": {}}
    print("\ntime outside the engine, per request (ms)")
    print(f"{'':22}{'direct':>8}{'via Relay':>11}{'Relay adds':>12}")
    for metric, label in (
        ("ttft_added_s", "before first token"),
        ("stream_added_s", "first->last token"),
    ):
        d, r = stats(measured["direct"][metric]), stats(measured["relay"][metric])
        report["outside_engine"][metric] = {"direct": d, "relay": r}
        report["relay_adds"][metric] = {q: r[q] - d[q] for q in d if None not in (d[q], r[q])}
        for q in ("p50", "p95", "p99"):
            added = None if None in (d[q], r[q]) else r[q] - d[q]
            print(f"{label + ' ' + q:22}{ms(d[q]):>8}{ms(r[q]):>11}{ms(added):>12}")
    for name, (_, _, s, _) in runs.items():
        measured_n = len(measured[name]["ttft_added_s"])
        report["requests"][name] = {"sent": s.sent, "errors": s.errors, "measured": measured_n}
    print(f"errors: direct {runs['direct'][2].errors}, via Relay {runs['relay'][2].errors}")
    count = after[1] - before[1]
    if count:
        mean = (after[0] - before[0]) / count
        report["server_side_mean_s"] = mean
        print(f"server side: {OVERHEAD_METRIC} mean {ms(mean)} ms over {count:.0f} requests")

    root = args.out / "_scratch" if args.scratch else args.out
    for name, (scenario, results, summary, _) in runs.items():
        out = write_run(root, scenario, results, [summary])
        if name == "relay":
            (out / "overhead.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
