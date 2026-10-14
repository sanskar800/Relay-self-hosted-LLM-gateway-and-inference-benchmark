"""Cost model: estimated £ per 1M output tokens from measured throughput and a list price.

    £ / 1M tokens = price per hour / (output tokens per second x 3600) x 1,000,000

The price is an assumption (bench/pricing.yaml, with source and date); the throughput is
measured. Assumes the GPU is fully used at that throughput for the whole hour.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

LABEL = "estimate: measured throughput x assumed public list price"


class Fx(BaseModel):
    model_config = ConfigDict(extra="forbid")
    usd_to_gbp: float = Field(gt=0)
    source: str
    retrieved: str


class GpuPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    usd_per_hour: float | None = Field(default=None, gt=0)
    source: str | None = None
    retrieved: str | None = None


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str
    fx: Fx
    gpus: dict[str, GpuPrice]

    def gbp_per_hour(self, gpu: str) -> float | None:
        price = self.gpus[gpu].usd_per_hour
        return None if price is None else price * self.fx.usd_to_gbp


def load_pricing(path: Path = Path("bench/pricing.yaml")) -> Pricing:
    return Pricing.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def cost_per_million_tokens(price_per_hour: float | None, tokens_per_s: float) -> float | None:
    """None when the price is unknown or nothing was produced (no division by zero)."""
    if price_per_hour is None or tokens_per_s <= 0:
        return None
    return price_per_hour / (tokens_per_s * 3600) * 1_000_000
