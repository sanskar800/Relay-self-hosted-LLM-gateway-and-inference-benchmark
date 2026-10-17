"""Benchmark CLI checks: probe settings, suggested rates, wrong-backend detection."""

from pathlib import Path

import pytest
import yaml
from bench.loadgen.__main__ import probe_scenario, suggested_rates, unexpected_backends
from bench.loadgen.config import Scenario
from bench.loadgen.metrics import Summary
from bench.loadgen.runner import RequestResult
from pydantic import ValidationError

SETUP = {
    "hardware": "RTX 4050 Laptop 6 GB",
    "engine": "llama.cpp b11459 (CUDA 12)",
    "model_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "quantization": "Q4_K_M",
    "server_args": "--parallel 4 --ctx-size 4096",
    "backend": "llamacpp-gpu",
}


def scenario(**extra) -> Scenario:
    raw = yaml.safe_load(Path("bench/scenarios/local-smoke.yaml").read_text(encoding="utf-8"))
    return Scenario.model_validate({**raw, **extra})


def result(backend: str | None) -> RequestResult:
    return RequestResult(level=1, index=0, scheduled_s=0, sent_s=0, status=200, backend=backend)


def summary(throughput: float) -> Summary:
    # Only throughput matters here; the other fields are filled with neutral values.
    fields = {f: 0 for f in Summary.__dataclass_fields__}
    return Summary(**{**fields, "level": 1, "throughput_rps": throughput})


def test_probe_is_a_short_closed_loop_with_the_same_workload() -> None:
    base = scenario(setup=SETUP)
    probe = probe_scenario(base)
    assert probe.name == "local-smoke-probe"
    assert probe.load.mode == "closed" and probe.load.concurrency == [1, 2, 4, 8]
    assert probe.workload == base.workload and probe.setup == base.setup


def test_suggested_rates_bracket_the_peak() -> None:
    capacity, rates = suggested_rates([summary(0.4), summary(1.2), summary(1.0)])
    assert capacity == 1.2
    assert rates == [0.3, 0.6, 0.9, 1.2, 1.5]  # light load up to past saturation


def test_a_run_served_by_another_backend_is_flagged() -> None:
    s = scenario(setup=SETUP)
    assert unexpected_backends(s, [result("llamacpp-gpu"), result(None)]) == set()
    assert unexpected_backends(s, [result("llamacpp-gpu"), result("llamacpp-cpu")]) == {
        "llamacpp-cpu"
    }
    assert unexpected_backends(scenario(), [result("anything")]) == set()  # smoke: no setup


def test_setup_must_name_the_backend() -> None:
    with pytest.raises(ValidationError):
        scenario(setup={k: v for k, v in SETUP.items() if k != "backend"})
