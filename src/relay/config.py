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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELAY_", env_file=".env", extra="ignore")

    # Env var is RELAY_CONFIG (an explicit alias, not RELAY_CONFIG_PATH).
    config_path: Path = Field(default=Path("config/relay.yaml"), validation_alias="RELAY_CONFIG")


class _Strict(BaseModel):
    # Config is written by us: an unknown key is a typo, so reject it.
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackendConfig(_Strict):
    engine: str
    url: HttpUrl
    connect_timeout_s: float = Field(default=5.0, gt=0)
    # For streams: the maximum silence between chunks, not the total duration.
    read_timeout_s: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def _known_engine(self) -> Self:
        if self.engine not in ENGINES:
            raise ValueError(f"unknown engine {self.engine!r}; expected one of {sorted(ENGINES)}")
        return self


class ModelConfig(_Strict):
    # Backend names in order of preference.
    backends: list[str] = Field(min_length=1)
    owned_by: str = "relay"


class RelayConfig(_Strict):
    backends: dict[str, BackendConfig] = Field(min_length=1)
    models: dict[str, ModelConfig] = Field(min_length=1)

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
