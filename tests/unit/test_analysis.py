"""The report's numbers: SQL over raw timestamps must agree with the harness, and the
report must regenerate identically (task 7.3)."""

import hashlib
import json
from pathlib import Path

import pytest
from bench.analysis import analyze

RUNS = sorted(Path("results").glob("run-*"))
pytestmark = pytest.mark.skipif(not RUNS, reason="no committed benchmark results")

# Fields computed by both bench/loadgen/metrics.py (summary.json) and queries.sql.
SHARED = [
    "sent", "ok", "errors", "error_rate", "window_s", "offered_rps", "throughput_rps",
    "output_tokens_per_s", "goodput_ratio", "ttft_p50", "ttft_p95", "ttft_p99", "itl_p50",
    "itl_p95", "itl_p99", "tpot_p50", "tpot_p95", "e2e_p50", "e2e_p95", "scheduling_lag_p99",
]  # fmt: skip


def test_sql_metrics_match_the_harness_for_every_level() -> None:
    runs = analyze.load_runs()
    con = analyze.connect(runs)
    sql = {(r["run_id"], r["level"]): r for r in analyze.levels(con, analyze.load_queries())}
    checked = 0
    for run in RUNS:
        for harness in json.loads((run / "summary.json").read_text(encoding="utf-8")):
            mine = sql[(run.name, harness["level"])]
            for field in SHARED:
                assert mine[field] == pytest.approx(harness[field], rel=1e-9, abs=1e-12), (
                    f"{run.name} level {harness['level']}: {field}"
                )
                checked += 1
    assert checked >= len(SHARED) * 3  # at least three levels really compared


def test_slo_limit_brackets_the_capacity() -> None:
    def level(rate: float, ttft: float, itl: float, err: float = 0.0) -> dict:
        return {"level": rate, "ttft_p95": ttft, "itl_p95": itl, "error_rate": err}

    levels = [level(1, 0.2, 0.02), level(2, 0.3, 0.05), level(3, 0.4, 0.12), level(4, 0.1, 0.01)]
    ok, miss = analyze.slo_limit(levels)
    assert (ok["level"], miss["level"]) == (2, 3)  # stops at the first miss, even if 4 "meets"
    ok, miss = analyze.slo_limit(levels[:2])
    assert ok["level"] == 2 and miss is None  # never missed: only a lower bound
    assert analyze.slo_limit([level(1, 0.2, 0.02, err=0.02)]) == (None, level(1, 0.2, 0.02, 0.02))


def test_report_regenerates_byte_for_byte() -> None:
    def digest() -> dict[str, str]:
        files = [analyze.REPORT, *sorted(analyze.CHARTS.glob("*.png"))]
        files += [r / "summary.md" for r in RUNS]
        return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}

    analyze.generate()
    first = digest()
    analyze.generate()
    assert digest() == first
