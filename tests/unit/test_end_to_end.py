"""Real OpenAI SDK -> Relay -> real HTTP backend -> scripted fake engine. Only the model is fake."""

from collections.abc import Iterator
from contextlib import contextmanager

import openai
from fastapi.testclient import TestClient

from relay.accounting.usage import UsageRecord
from relay.main import create_app
from tests.fake_engine import FakeEngine
from tests.unit.fakes import make_config

MESSAGES = [{"role": "user", "content": "hi"}]


@contextmanager
def stack(engine: FakeEngine) -> Iterator[tuple[openai.OpenAI, list[UsageRecord]]]:
    records: list[UsageRecord] = []
    app = create_app(
        config=make_config("m", "llamacpp-cpu"),
        transport=engine.transport(),
        usage_sink=records.append,
    )
    with TestClient(app) as http:
        yield (
            openai.OpenAI(
                base_url="http://testserver/v1", api_key="x", http_client=http, max_retries=0
            ),
            records,
        )


def test_non_streaming_round_trip_and_usage() -> None:
    engine = FakeEngine(tokens=["Hel", "lo", "!"], prompt_tokens=7)
    with stack(engine) as (client, records):
        reply = client.chat.completions.create(model="m", messages=MESSAGES)
    assert reply.choices[0].message.content == "Hello!"
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (7, 3)
    assert records == [UsageRecord("m", "llamacpp-cpu", False, 200, 7, 3, "engine")]


def test_stream_without_include_usage_still_accounts_exactly() -> None:
    engine = FakeEngine(tokens=["a", "b", "c", "d"], prompt_tokens=11)
    with stack(engine) as (client, records):
        chunks = list(client.chat.completions.create(model="m", messages=MESSAGES, stream=True))

    # The client sees a normal stream: text, a finish_reason, and no usage chunk.
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "abcd"
    assert all(c.choices for c in chunks)  # no choices: [] chunk leaked to the client
    assert chunks[-1].choices[0].finish_reason == "stop"
    # Relay asked the engine for usage, and the engine only reports it when asked...
    assert engine.requests[0]["stream_options"] == {"include_usage": True}
    # ...so the exact count recorded here proves the injection worked end to end.
    assert records[0].source == "engine"
    assert (records[0].prompt_tokens, records[0].completion_tokens) == (11, 4)


def test_stream_with_include_usage_delivers_it_to_the_client() -> None:
    engine = FakeEngine(tokens=["x", "y"], prompt_tokens=5)
    with stack(engine) as (client, records):
        chunks = list(
            client.chat.completions.create(
                model="m",
                messages=MESSAGES,
                stream=True,
                stream_options={"include_usage": True},
            )
        )
    assert chunks[-1].choices == []
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (5, 2)
    assert records[0].completion_tokens == 2


def test_unknown_parameters_reach_the_engine() -> None:
    engine = FakeEngine()
    with stack(engine) as (client, _):
        client.chat.completions.create(
            model="m", messages=MESSAGES, seed=42, extra_body={"top_k": 40, "min_p": 0.05}
        )
    sent = engine.requests[0]
    assert (sent["seed"], sent["top_k"], sent["min_p"]) == (42, 40, 0.05)


def test_models_endpoint_through_the_sdk() -> None:
    with stack(FakeEngine()) as (client, _):
        assert [m.id for m in client.models.list()] == ["m"]
