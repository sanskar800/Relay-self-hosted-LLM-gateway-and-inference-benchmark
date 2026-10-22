"""--auto-rates end to end (the unattended Colab path): probe, then the open loop at the
probe's suggested rates, with the probe kept as evidence. Short durations; fake engine."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from bench.loadgen import __main__ as cli
from bench.loadgen import memory

from tests.engine_server import serve_engine
from tests.fake_engine import FakeEngine

pytestmark = pytest.mark.integration


def test_auto_rates_probes_then_runs_at_the_suggested_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "PROBE_LOAD",
        {"mode": "closed", "concurrency": [1, 2], "duration_s": 1.5, "warmup_s": 0},
    )
    # The laptop's free RAM is not under test here.
    monkeypatch.setattr(cli, "available_gib", lambda: 8.0)
    monkeypatch.setattr(memory, "MIN_AVAILABLE_DURING_RUN_GIB", 0.0)

    engine = FakeEngine(tokens=["t"] * 5, first_event_delay_s=0.05, token_delay_s=0.01)
    with serve_engine(engine) as url:
        raw = yaml.safe_load(Path("bench/scenarios/local-smoke.yaml").read_text(encoding="utf-8"))
        raw["target"] = {"base_url": f"{url}/v1", "model": "m"}
        raw["load"].update(arrival_rates_rps=[99.0], duration_s=2.0, warmup_s=0.5)
        scenario_file = tmp_path / "s.yaml"
        scenario_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
        cli.main([str(scenario_file), "--auto-rates", "--out", str(tmp_path / "results")])

    results = tmp_path / "results"
    (probe,) = (results / "dev").iterdir()  # the probe is kept as evidence
    (run,) = [p for p in results.iterdir() if p.name.startswith("run-")]
    env = json.loads((run / "env.json").read_text())
    used = yaml.safe_load((run / "config.yaml").read_text())["load"]["arrival_rates_rps"]

    peak = env["rates_from_probe"]["peak_throughput_rps"]
    assert env["rates_from_probe"]["probe_run"] == probe.name
    assert used != [99.0]  # the scenario's own rates were replaced ...
    assert used == [round(peak * f, 2) for f in cli.PROBE_FRACTIONS]  # ... by the probe's
    assert env["invalid_reasons"] == []
    rows = pq.read_table(run / "results.parquet").to_pylist()
    assert {r["level"] for r in rows} == set(used) and all(r["ok"] for r in rows)
