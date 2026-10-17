"""Results folder per run: config.yaml, results.parquet (one row per request), summary.json,
env.json. Everything a later analysis needs, so no benchmark has to be re-run for a chart.
"""

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from bench.loadgen.config import Scenario
from bench.loadgen.metrics import Summary, tpot, ttft
from bench.loadgen.runner import RequestResult

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("scenario", pa.string()),
        ("mode", pa.string()),
        ("level", pa.float64()),
        ("index", pa.int32()),
        ("in_warmup", pa.bool_()),
        ("scheduled_s", pa.float64()),
        ("sent_s", pa.float64()),
        ("first_token_s", pa.float64()),
        ("end_s", pa.float64()),
        ("ttft_s", pa.float64()),
        ("tpot_s", pa.float64()),
        ("e2e_s", pa.float64()),
        ("token_times_s", pa.list_(pa.float64())),
        ("status", pa.int32()),
        ("ok", pa.bool_()),
        ("error", pa.string()),
        ("target_prompt_tokens", pa.int32()),
        ("prompt_tokens", pa.int32()),
        ("max_tokens", pa.int32()),
        ("completion_tokens", pa.int32()),
        ("backend", pa.string()),
        ("engine_fingerprint", pa.string()),
        ("request_id", pa.string()),
    ]
)


def run_dir(root: Path, scenario: Scenario, now: datetime | None = None) -> Path:
    now = now or datetime.now(UTC)
    return root / f"run-{now:%Y-%m-%d}-{scenario.name}-{now:%H%M%S}"


def _rows(run_id: str, scenario: Scenario, results: list[RequestResult]) -> list[dict]:
    rows = []
    for r in results:
        e2e = r.end_s - r.sent_s if r.end_s is not None and r.sent_s is not None else None
        rows.append(
            {
                "run_id": run_id,
                "scenario": scenario.name,
                "mode": scenario.load.mode,
                "level": r.level,
                "index": r.index,
                "in_warmup": r.in_warmup,
                "scheduled_s": r.scheduled_s,
                "sent_s": r.sent_s,
                "first_token_s": r.first_token_s,
                "end_s": r.end_s,
                "ttft_s": ttft(r),
                "tpot_s": tpot(r),
                "e2e_s": e2e,
                "token_times_s": r.token_times_s,
                "status": r.status,
                "ok": r.ok,
                "error": r.error,
                "target_prompt_tokens": r.target_prompt_tokens,
                "prompt_tokens": r.prompt_tokens,
                "max_tokens": r.max_tokens,
                "completion_tokens": r.completion_tokens,
                "backend": r.backend,
                "engine_fingerprint": r.engine_fingerprint,
                "request_id": r.request_id,
            }
        )
    return rows


def _git() -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _gpu() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        name, driver, memory = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "driver": driver, "memory": memory})
    return gpus or None


def environment(
    scenario: Scenario, results: list[RequestResult], prompt_salt: str | None = None
) -> dict:
    versions = {}
    for pkg in ("relay", "httpx", "pyarrow"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "git": _git(),
        "host": {
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        # GPUs on the machine running the HARNESS; for a remote target (e.g. Colab) the
        # engine fingerprints below identify what served the requests.
        "gpu": _gpu(),
        "packages": versions,
        "target": {"base_url": str(scenario.target.base_url), "model": scenario.target.model},
        # Prompt openings are salted per run (bench.loadgen.workload.stream); with this
        # value the exact prompts can be regenerated.
        "prompt_salt": prompt_salt,
        "engine_fingerprints": sorted(
            {r.engine_fingerprint for r in results if r.engine_fingerprint}
        ),
        "backends_seen": sorted({r.backend for r in results if r.backend}),
    }


def _write(path: Path, text: str) -> None:
    # LF on every OS: results are committed, and CRLF would differ between machines.
    path.write_text(text, encoding="utf-8", newline="\n")


def write_run(
    root: Path,
    scenario: Scenario,
    results: list[RequestResult],
    summaries: list[Summary],
    now: datetime | None = None,
    prompt_salt: str | None = None,
    extra_env: dict | None = None,
) -> Path:
    """`extra_env` adds measured facts about the run to env.json (e.g. host memory)."""
    out = run_dir(root, scenario, now)
    out.mkdir(parents=True, exist_ok=False)
    _write(out / "config.yaml", yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False))
    table = pa.Table.from_pylist(_rows(out.name, scenario, results), schema=SCHEMA)
    pq.write_table(table, out / "results.parquet")
    _write(out / "summary.json", json.dumps([dataclasses.asdict(s) for s in summaries], indent=2))
    env = {**environment(scenario, results, prompt_salt), **(extra_env or {})}
    _write(out / "env.json", json.dumps(env, indent=2))
    return out
