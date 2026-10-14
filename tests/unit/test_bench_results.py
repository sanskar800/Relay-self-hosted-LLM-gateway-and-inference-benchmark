"""Run folder layout, Parquet round-trip, DuckDB over many runs, and the cost formula."""

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest
import yaml
from bench.analysis.cost import cost_per_million_tokens, load_pricing
from bench.loadgen.config import load_scenario
from bench.loadgen.metrics import summarise
from bench.loadgen.results import write_run
from bench.loadgen.runner import RequestResult

SCENARIO = Path("bench/scenarios/local-smoke.yaml")


def results() -> list[RequestResult]:
    ok = RequestResult(
        level=0.5, index=0, scheduled_s=10.0, sent_s=10.0, status=200,
        first_token_s=10.5, token_times_s=[10.5, 10.6, 10.7], end_s=10.7,
        prompt_tokens=150, completion_tokens=3, target_prompt_tokens=120, max_tokens=3,
        backend="llamacpp", engine_fingerprint="b1-abc", request_id="r-1",
    )  # fmt: skip
    failed = RequestResult(
        level=0.5, index=1, scheduled_s=11.0, sent_s=11.0, status=503, error="busy", end_s=11.1
    )
    return [ok, failed]


def test_write_run_creates_a_complete_folder(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    rs = results()
    now = datetime(2026, 10, 14, 9, 30, 5, tzinfo=UTC)
    out = write_run(tmp_path, scenario, rs, [summarise(rs, 0, scenario.slo)], now=now)

    assert out.name == "run-2026-10-14-local-smoke-093005"
    assert sorted(p.name for p in out.iterdir()) == [
        "config.yaml", "env.json", "results.parquet", "summary.json",
    ]  # fmt: skip
    # The saved config reloads as the same scenario: the run is reproducible from its folder.
    assert yaml.safe_load((out / "config.yaml").read_text())["name"] == scenario.name
    assert json.loads((out / "summary.json").read_text())[0]["sent"] == 2
    env = json.loads((out / "env.json").read_text())
    assert env["engine_fingerprints"] == ["b1-abc"] and env["backends_seen"] == ["llamacpp"]
    assert "sha" in env["git"]


def test_parquet_round_trip_keeps_per_request_detail(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    out = write_run(tmp_path, scenario, results(), [])
    rows = pq.read_table(out / "results.parquet").to_pylist()
    assert len(rows) == 2
    first, second = rows
    assert first["ttft_s"] == pytest.approx(0.5)
    assert first["tpot_s"] == pytest.approx(0.1)
    assert first["token_times_s"] == [10.5, 10.6, 10.7]
    assert first["ok"] and first["run_id"] == out.name
    assert not second["ok"] and second["status"] == 503 and second["ttft_s"] is None


def test_duckdb_queries_across_runs(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    for second in (1, 2):
        now = datetime(2026, 10, 14, 9, 0, second, tzinfo=UTC)
        write_run(tmp_path, scenario, results(), [], now=now)
    glob = (tmp_path / "*" / "results.parquet").as_posix()
    runs, rows, ok = duckdb.sql(
        f"select count(distinct run_id), count(*), count(*) filter (where ok) from '{glob}'"
    ).fetchone()
    assert (runs, rows, ok) == (2, 4, 2)


def test_existing_run_folder_is_never_overwritten(tmp_path: Path) -> None:
    scenario = load_scenario(SCENARIO)
    now = datetime(2026, 10, 14, tzinfo=UTC)
    write_run(tmp_path, scenario, results(), [], now=now)
    with pytest.raises(FileExistsError):
        write_run(tmp_path, scenario, results(), [], now=now)


# --- cost model -----------------------------------------------------------------------


def test_cost_formula() -> None:
    # £1/h at 1000 tok/s = 3.6M tok/h -> £0.2778 per 1M tokens.
    assert cost_per_million_tokens(1.0, 1000) == pytest.approx(1 / 3.6)
    # Doubling throughput halves the cost.
    assert cost_per_million_tokens(1.0, 2000) == pytest.approx(
        cost_per_million_tokens(1.0, 1000) / 2
    )


def test_cost_is_none_without_a_price_or_throughput() -> None:
    assert cost_per_million_tokens(None, 1000) is None  # unknown price -> TBD, never guessed
    assert cost_per_million_tokens(1.0, 0) is None


def test_pricing_file_has_sources_and_converts_to_gbp() -> None:
    pricing = load_pricing()
    assert pricing.currency == "GBP"
    for gpu in pricing.gpus.values():
        if gpu.usd_per_hour is not None:  # every price states where it came from and when
            assert gpu.source and gpu.retrieved
    assert pricing.gbp_per_hour("t4") == pytest.approx(0.615 * 0.747)
    assert pricing.gbp_per_hour("rtx-4050-laptop") is None
