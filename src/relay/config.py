"""Runtime settings, read from RELAY_* environment variables (or a .env file)."""

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RELAY_", env_file=".env", extra="ignore")

    # OpenAI-compatible backend. A YAML model -> backends map replaces this on Day 2.
    backend_url: HttpUrl = HttpUrl("http://localhost:8081")

    # LLM responses take seconds, so the read timeout is long. Day 4 splits these
    # into connect / first-byte / idle / total timeouts per backend.
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 120.0
