"""Settings (environment) and the model/backend routing config (YAML).

Environment variables (RELAY_*) hold per-deployment knobs; config/relay.yaml holds the
routing table, which is validated strictly at startup so typos fail fast.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from relay.backends import ENGINES
from relay.backends.base import Timeouts
from relay.routing.circuit_breaker import BreakerPolicy
from relay.routing.retry import RetryPolicy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELAY_", env_file=".env", extra="ignore")

    # Env var is RELAY_CONFIG (an explicit alias, not RELAY_CONFIG_PATH).
    config_path: Path = Field(default=Path("config/relay.yaml"), validation_alias="RELAY_CONFIG")
    pricing_path: Path = Field(
        default=Path("config/pricing.yaml"), validation_alias="RELAY_PRICING"
    )
    # `python -m relay.serve` (production-style server with graceful shutdown).
    host: str = "127.0.0.1"
    port: int = 8000
    shutdown_grace_s: float = Field(default=5.0, ge=0)  # not ready, still accepting
    drain_timeout_s: float = Field(default=30.0, gt=0)  # then wait this long for in-flight
    # OTLP/HTTP collector for traces (e.g. Jaeger at http://127.0.0.1:4318). Unset = off.
    otel_endpoint: str | None = Field(default=None, validation_alias="RELAY_OTEL_ENDPOINT")
    # Defaults match the local Docker Compose stack (see .env.example).
    database_url: str = "postgresql://relay:relay@127.0.0.1:5433/relay"
    redis_url: str = "redis://127.0.0.1:6379/0"


class _Strict(BaseModel):
    # Config is written by us: an unknown key is a typo, so reject it.
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackendConfig(_Strict):
    engine: str
    url: HttpUrl
    # Timeouts (seconds); see relay.backends.base.Timeouts for what each one bounds.
    connect_timeout_s: float = Field(default=5.0, gt=0)
    first_token_timeout_s: float = Field(default=60.0, gt=0)
    idle_timeout_s: float = Field(default=30.0, gt=0)
    total_timeout_s: float = Field(default=300.0, gt=0)
    # Reuse connections to this backend. Unset = the engine's default (llama.cpp: off,
    # docs/findings.md F7; others: on).
    keep_alive: bool | None = None

    @model_validator(mode="after")
    def _known_engine(self) -> Self:
        if self.engine not in ENGINES:
            raise ValueError(f"unknown engine {self.engine!r}; expected one of {sorted(ENGINES)}")
        return self

    @model_validator(mode="after")
    def _consistent_timeouts(self) -> Self:
        for name in ("connect_timeout_s", "first_token_timeout_s", "idle_timeout_s"):
            if getattr(self, name) > self.total_timeout_s:
                raise ValueError(f"{name} cannot exceed total_timeout_s")
        return self

    def timeouts(self) -> Timeouts:
        return Timeouts(
            connect=self.connect_timeout_s,
            first_token=self.first_token_timeout_s,
            idle=self.idle_timeout_s,
            total=self.total_timeout_s,
        )


class ModelConfig(_Strict):
    # Backend names in order of preference.
    backends: list[str] = Field(min_length=1)
    owned_by: str = "relay"


class LimitsConfig(_Strict):
    """Global request limits. Per-tenant caps can only be stricter than these."""

    max_body_bytes: int = Field(default=1_048_576, gt=0)  # 1 MB (requirements brief)
    max_messages: int = Field(default=256, gt=0)
    # Ceiling for max_tokens / max_completion_tokens when the client sends them. If the
    # client omits them, the backend's context window bounds generation instead.
    max_tokens: int = Field(default=4096, gt=0)


class RetryConfig(_Strict):
    """Retries before the first token (ADR-003)."""

    max_attempts: int = Field(default=3, ge=1, le=10)  # including the first try
    base_delay_s: float = Field(default=0.1, ge=0)
    max_delay_s: float = Field(default=1.0, ge=0)

    def policy(self) -> RetryPolicy:
        return RetryPolicy(self.max_attempts, self.base_delay_s, self.max_delay_s)


class BreakerConfig(_Strict):
    """Circuit breaker per backend."""

    failure_threshold: int = Field(default=3, ge=1)  # consecutive failures to open
    open_duration_s: float = Field(default=10.0, gt=0)  # cool-down before a probe

    def policy(self) -> BreakerPolicy:
        return BreakerPolicy(self.failure_threshold, self.open_duration_s)


class RelayConfig(_Strict):
    backends: dict[str, BackendConfig] = Field(min_length=1)
    models: dict[str, ModelConfig] = Field(min_length=1)
    limits: LimitsConfig = LimitsConfig()
    retry: RetryConfig = RetryConfig()
    breaker: BreakerConfig = BreakerConfig()

    @model_validator(mode="after")
    def _references_exist(self) -> Self:
        for model, cfg in self.models.items():
            unknown = [name for name in cfg.backends if name not in self.backends]
            if unknown:
                raise ValueError(f"model {model!r} refers to undefined backends: {unknown}")
            if len(set(cfg.backends)) != len(cfg.backends):
                raise ValueError(f"model {model!r} lists a backend more than once")
        return self


def load_config(path: Path) -> RelayConfig:
    """Read and validate the routing config. Raises with a clear message if invalid."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"config file not found: {path} (set RELAY_CONFIG)") from None
    return RelayConfig.model_validate(raw)


class ModelPrice(_Strict):
    # £ per 1M tokens. None = not yet known (TBD); cost is then recorded as NULL.
    input_gbp_per_1m: float | None = Field(default=None, ge=0)
    output_gbp_per_1m: float | None = Field(default=None, ge=0)


class PricingConfig(_Strict):
    basis: str
    models: dict[str, ModelPrice] = {}

    def cost_gbp(self, model: str, prompt: int | None, completion: int | None) -> float | None:
        """Estimated cost, or None if a rate or a token count is unknown."""
        price = self.models.get(model)
        if price is None or None in (
            price.input_gbp_per_1m,
            price.output_gbp_per_1m,
            prompt,
            completion,
        ):
            return None
        return (prompt * price.input_gbp_per_1m + completion * price.output_gbp_per_1m) / 1e6


def load_pricing(path: Path) -> PricingConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"pricing file not found: {path} (set RELAY_PRICING)") from None
    return PricingConfig.model_validate(raw)
