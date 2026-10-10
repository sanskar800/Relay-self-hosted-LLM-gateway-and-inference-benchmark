from pathlib import Path

import pytest
from pydantic import ValidationError

from relay.config import RelayConfig, Settings, load_config

VALID = {
    "backends": {
        "cpu": {"engine": "llamacpp", "url": "http://127.0.0.1:8081"},
        "gpu": {"engine": "llamacpp", "url": "http://127.0.0.1:8082", "read_timeout_s": 30},
    },
    "models": {"qwen": {"backends": ["gpu", "cpu"]}},
}


def test_repository_config_is_valid() -> None:
    config = load_config(Path("config/relay.yaml"))
    assert "qwen2.5-1.5b-instruct" in config.models
    for model in config.models.values():
        assert all(name in config.backends for name in model.backends)


def test_valid_config_keeps_order_and_defaults() -> None:
    config = RelayConfig.model_validate(VALID)
    assert config.models["qwen"].backends == ["gpu", "cpu"]  # order = preference
    assert config.backends["cpu"].connect_timeout_s == 5.0
    assert config.backends["gpu"].read_timeout_s == 30


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["backends"]["cpu"].update(engine="tgi"), "unknown engine 'tgi'"),
        (lambda c: c["models"]["qwen"].update(backends=["gpu", "cpuu"]), "undefined backends"),
        (lambda c: c["models"]["qwen"].update(backends=["cpu", "cpu"]), "more than once"),
        (lambda c: c["models"]["qwen"].update(backends=[]), "at least 1 item"),
        (lambda c: c["backends"]["cpu"].update(url="not a url"), "URL"),
        (lambda c: c["backends"]["cpu"].update(read_timeout_s=0), "greater than 0"),
        (lambda c: c["backends"]["cpu"].update(timeout=5), "Extra inputs are not permitted"),
        (lambda c: c.update(model={}), "Extra inputs are not permitted"),
        (lambda c: c.update(models={}), "at least 1 item"),
    ],
)
def test_invalid_config_is_rejected_with_a_clear_message(change, message: str) -> None:
    config = {
        "backends": {name: dict(cfg) for name, cfg in VALID["backends"].items()},
        "models": {name: dict(cfg) for name, cfg in VALID["models"].items()},
    }
    change(config)
    with pytest.raises(ValidationError, match=message):
        RelayConfig.model_validate(config)


def test_missing_file_names_the_env_var(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RELAY_CONFIG"):
        load_config(tmp_path / "absent.yaml")


def test_config_path_comes_from_relay_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_CONFIG", "elsewhere/relay.yaml")
    assert Settings().config_path == Path("elsewhere/relay.yaml")
