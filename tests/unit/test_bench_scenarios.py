"""Every benchmark scenario is valid and measures the same workload (ADR-001)."""

from pathlib import Path

import pytest
from bench.loadgen.config import load_scenario

SCENARIOS = sorted(Path("bench/scenarios").glob("*.yaml"))
BENCHMARKS = [p for p in SCENARIOS if load_scenario(p).setup is not None]


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_is_valid_and_named_after_its_file(path: Path) -> None:
    assert load_scenario(path).name == path.stem


def test_benchmarks_share_one_workload_and_slo() -> None:
    # Configurations are comparable only if they served the same requests and were held
    # to the same targets; only hardware, engine, model and rates may differ.
    assert len(BENCHMARKS) >= 2
    first = load_scenario(BENCHMARKS[0])
    for path in BENCHMARKS[1:]:
        s = load_scenario(path)
        assert s.workload == first.workload, path.name
        assert s.slo == first.slo, path.name
        assert (s.load.duration_s, s.load.warmup_s) == (
            first.load.duration_s,
            first.load.warmup_s,
        ), path.name


@pytest.mark.parametrize("path", BENCHMARKS, ids=lambda p: p.stem)
def test_benchmarks_use_the_bench_tenant_and_an_open_loop(path: Path) -> None:
    s = load_scenario(path)
    assert s.target.api_key_env == "RELAY_BENCH_API_KEY"
    assert s.load.mode == "open"  # closed loops hide queueing (findings F5)
