"""Overhead = harness-observed time minus the engine's own recorded time, per request."""

import pytest
from bench.loadgen.runner import RequestResult
from bench.loadgen.workload import RequestSpec
from bench.overhead import outside_engine, relay_config

from relay.config import RelayConfig


def spec(index: int, prompt: str) -> RequestSpec:
    return RequestSpec(index=index, prompt=prompt, target_prompt_tokens=10, max_tokens=3)


def test_engine_jitter_cancels_out() -> None:
    specs = [spec(0, "a"), spec(1, "b")]
    # Engine clock: request "a" took 0.10 s to its first token, "b" 0.13 s (jitter).
    # Timelines arrive in engine order, not request order: matched by prompt.
    timelines = [
        {"prompt": "b", "received": 5.0, "tokens": [5.13, 5.15, 5.17]},
        {"prompt": "a", "received": 1.0, "tokens": [1.10, 1.12, 1.14]},
    ]
    # The harness saw each 4 ms later for the first token, and the stream 1 ms slower.
    results = [
        RequestResult(level=1, index=0, scheduled_s=0, sent_s=0.0, status=200,
                      first_token_s=0.104, token_times_s=[0.104, 0.124, 0.145], end_s=0.15),
        RequestResult(level=1, index=1, scheduled_s=4, sent_s=4.0, status=200,
                      first_token_s=4.134, token_times_s=[4.134, 4.154, 4.175], end_s=4.18),
    ]  # fmt: skip
    measured = outside_engine(results, specs, timelines)
    assert measured["ttft_added_s"] == pytest.approx([0.004, 0.004])
    assert measured["stream_added_s"] == pytest.approx([0.001, 0.001])


def test_warmup_and_failed_requests_are_left_out() -> None:
    specs = [spec(0, "a"), spec(1, "b")]
    results = [
        RequestResult(level=1, index=0, scheduled_s=0, sent_s=0, status=200, in_warmup=True,
                      first_token_s=0.1, token_times_s=[0.1], end_s=0.1),
        RequestResult(level=1, index=1, scheduled_s=1, sent_s=1, status=503, end_s=1.01),
    ]  # fmt: skip
    assert outside_engine(results, specs, []) == {"ttft_added_s": [], "stream_added_s": []}


def test_generated_relay_config_is_valid() -> None:
    config = RelayConfig.model_validate(relay_config("http://127.0.0.1:9999", "m"))
    assert config.models["m"].backends == ["fake-engine"]
