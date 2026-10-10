"""Relay must work with the official OpenAI SDK, unchanged, by pointing base_url at it."""

import openai
import pytest

pytestmark = pytest.mark.contract


def test_chat_completion_round_trip(client: openai.OpenAI, model: str) -> None:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=10,
        temperature=0,
    )

    # The SDK parsed the response into its own typed objects: the wire format is right.
    assert isinstance(completion, openai.types.chat.ChatCompletion)
    choice = completion.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str) and choice.message.content.strip()
    assert choice.finish_reason in {"stop", "length"}

    # Token usage is what accounting will rely on.
    assert completion.usage is not None
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.completion_tokens > 0
    assert completion.usage.total_tokens == (
        completion.usage.prompt_tokens + completion.usage.completion_tokens
    )


def test_max_tokens_is_respected(client: openai.OpenAI, model: str) -> None:
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Write a long story about a lighthouse."}],
        max_tokens=5,
        temperature=0,
    )
    assert completion.usage.completion_tokens <= 5
    assert completion.choices[0].finish_reason == "length"


def test_streaming_yields_incremental_chunks(client: openai.OpenAI, model: str) -> None:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5, separated by spaces."}],
        max_tokens=30,
        temperature=0,
        stream=True,
    )
    chunks = list(stream)

    assert all(isinstance(c, openai.types.chat.ChatCompletionChunk) for c in chunks)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert "1" in text and "5" in text
    content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]
    assert len(content_chunks) > 1  # token by token, not one buffered blob
    assert chunks[-1].choices[0].finish_reason in {"stop", "length"}


def test_streaming_usage_chunk_with_include_usage(client: openai.OpenAI, model: str) -> None:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi."}],
        max_tokens=10,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    last = chunks[-1]
    assert last.choices == []  # the usage chunk has no choices
    assert last.usage is not None and last.usage.completion_tokens > 0


def test_validation_error_is_parsed_by_the_sdk(client: openai.OpenAI, model: str) -> None:
    with pytest.raises(openai.BadRequestError) as excinfo:
        client.chat.completions.create(
            model=model, messages=[{"role": "robot", "content": "hi"}], max_tokens=5
        )
    assert excinfo.value.status_code == 400
    assert excinfo.value.body["param"] == "messages.0.role"


def test_models_list_and_retrieve(client: openai.OpenAI, model: str) -> None:
    listed = {m.id for m in client.models.list()}
    assert model in listed
    assert client.models.retrieve(model).id == model


def test_unknown_model_raises_not_found(client: openai.OpenAI) -> None:
    with pytest.raises(openai.NotFoundError) as excinfo:
        client.chat.completions.create(
            model="no-such-model", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
    assert excinfo.value.code == "model_not_found"
