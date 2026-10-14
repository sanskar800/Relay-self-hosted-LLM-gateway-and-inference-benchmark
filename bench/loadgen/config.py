"""Benchmark scenarios: one YAML file = one reproducible experiment.

Validated strictly (unknown keys rejected), like Relay's own config: a typo must fail the
run, not silently change what is measured.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Target(_Strict):
    # Relay (http://127.0.0.1:8000/v1) or a backend directly (e.g. http://127.0.0.1:8081/v1).
    base_url: HttpUrl
    model: str = Field(min_length=1)
    # Name of the environment variable holding the API key, never the key itself.
    api_key_env: str | None = None


class LengthDistribution(_Strict):
    """Log-normal lengths clipped to [min, max]: most near the median, a long right tail."""

    distribution: Literal["lognormal"] = "lognormal"
    median: int = Field(gt=0)
    sigma: float = Field(ge=0, le=3)  # spread of log(length); 0 = every request the same
    min: int = Field(ge=1)
    max: int = Field(ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.min <= self.median <= self.max:
            raise ValueError("expected min <= median <= max")
        return self


class Workload(_Strict):
    seed: int = 42
    prompt_tokens: LengthDistribution
    output_tokens: LengthDistribution
    # Fixed output lengths (max_tokens + ignore_eos), so engines are compared like for like.
    ignore_eos: bool = True
    # A unique opening per prompt, so the engine cannot reuse cached prefixes across
    # requests the way identical benchmark prompts would let it.
    unique_prefix: bool = True


class Load(_Strict):
    mode: Literal["open", "closed"] = "open"
    # Open loop: one run per rate, requests arrive as a Poisson process at this rate.
    arrival_rates_rps: list[float] = Field(default_factory=list)
    # Closed loop: N workers, each sending its next request when the previous finishes.
    concurrency: list[int] = Field(default_factory=list)
    duration_s: float = Field(gt=0)  # per rate / concurrency level
    # Requests that START during the warm-up are sent but excluded from the metrics.
    warmup_s: float = Field(default=0, ge=0)
    request_timeout_s: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def _levels(self) -> Self:
        if self.mode == "open" and (not self.arrival_rates_rps or min(self.arrival_rates_rps) <= 0):
            raise ValueError("open loop needs arrival_rates_rps, all > 0")
        if self.mode == "closed" and (not self.concurrency or min(self.concurrency) < 1):
            raise ValueError("closed loop needs concurrency levels, all >= 1")
        if self.warmup_s >= self.duration_s:
            raise ValueError("warmup_s must be shorter than duration_s")
        return self


class Slo(_Strict):
    """Per-request targets (docs/brief.md). A request meeting both counts towards goodput."""

    ttft_s: float = Field(default=1.0, gt=0)
    tpot_s: float = Field(default=0.1, gt=0)  # time per output token after the first


class Scenario(_Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str = ""
    target: Target
    load: Load
    workload: Workload
    slo: Slo = Slo()


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
