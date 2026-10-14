"""Benchmark scenarios and the seeded workload mix."""

import math
import statistics
from pathlib import Path

import pytest
from bench.loadgen.config import LengthDistribution, Scenario, Workload, load_scenario
from bench.loadgen.workload import TOKENS_PER_WORD, generate, request_body, sample_length
from pydantic import ValidationError

VALID = {
    "name": "t",
    "target": {"base_url": "http://127.0.0.1:8000/v1", "model": "m"},
    "load": {"arrival_rates_rps": [1.0], "duration_s": 10, "warmup_s": 2},
    "workload": {
        "prompt_tokens": {"median": 100, "sigma": 0.5, "min": 10, "max": 1000},
        "output_tokens": {"median": 50, "sigma": 0.4, "min": 5, "max": 500},
    },
}


@pytest.mark.parametrize("path", sorted(Path("bench/scenarios").glob("*.yaml")))
def test_repository_scenarios_are_valid(path: Path) -> None:
    load_scenario(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s["load"].update(arrival_rates_rps=[]), "arrival_rates_rps"),
        (lambda s: s["load"].update(arrival_rates_rps=[0]), "arrival_rates_rps"),
        (lambda s: s["load"].update(mode="closed"), "concurrency"),
        (lambda s: s["load"].update(warmup_s=10), "warmup_s"),
        (lambda s: s["workload"]["prompt_tokens"].update(min=200), "min <= median <= max"),
        (lambda s: s.update(name="Bad Name"), "pattern"),
        (lambda s: s["load"].update(duraton_s=5), "Extra inputs"),
        (lambda s: s["target"].update(api_key="rk_secret"), "Extra inputs"),  # keys never inline
    ],
)
def test_invalid_scenarios_are_rejected(change, message: str) -> None:
    scenario = {k: (dict(v) if isinstance(v, dict) else v) for k, v in VALID.items()}
    scenario["workload"] = {
        "prompt_tokens": dict(VALID["workload"]["prompt_tokens"]),
        "output_tokens": dict(VALID["workload"]["output_tokens"]),
    }
    change(scenario)
    with pytest.raises(ValidationError, match=message):
        Scenario.model_validate(scenario)


def test_same_seed_same_workload_different_seed_different() -> None:
    w = Scenario.model_validate(VALID).workload
    assert generate(w, 50) == generate(w, 50)
    other = w.model_copy(update={"seed": 7})
    assert generate(other, 50) != generate(w, 50)


def test_lengths_follow_the_documented_lognormal() -> None:
    import random

    dist = LengthDistribution(median=200, sigma=0.5, min=1, max=100_000)  # clipping inactive
    rng = random.Random(1)
    samples = sorted(sample_length(dist, rng) for _ in range(5000))
    q = statistics.quantiles(samples, n=10)  # deciles
    assert statistics.median(samples) == pytest.approx(200, rel=0.05)
    # Lognormal: the p-th quantile is median * exp(sigma * z_p); z_0.9 = 1.2816.
    assert q[8] == pytest.approx(200 * math.exp(0.5 * 1.2816), rel=0.06)
    assert q[0] == pytest.approx(200 * math.exp(-0.5 * 1.2816), rel=0.06)


def test_lengths_are_clipped_to_min_and_max() -> None:
    import random

    dist = LengthDistribution(median=100, sigma=2.0, min=50, max=150)
    rng = random.Random(3)
    samples = [sample_length(dist, rng) for _ in range(2000)]
    assert min(samples) == 50 and max(samples) == 150


def test_prompts_have_unique_openings_and_planned_size() -> None:
    w = Scenario.model_validate(VALID).workload
    specs = generate(w, 200)
    openings = {s.prompt.split("]")[0] for s in specs}
    assert len(openings) == 200  # no shared cacheable prefix between requests
    for s in specs[:20]:
        words = len(s.prompt.split("\n", 1)[1].split())
        assert words == max(1, round(s.target_prompt_tokens / TOKENS_PER_WORD))

    same = Workload.model_validate({**VALID["workload"], "unique_prefix": False})
    assert all(s.prompt.startswith("Continue") for s in generate(same, 10))


def test_request_body_fixes_output_length_and_asks_for_usage() -> None:
    w = Scenario.model_validate(VALID).workload
    spec = generate(w, 1)[0]
    body = request_body(spec, "m", w)
    assert body["max_tokens"] == spec.max_tokens and body["ignore_eos"] is True
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.0
