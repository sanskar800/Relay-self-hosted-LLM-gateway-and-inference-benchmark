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


def test_error_body_is_parsed_by_the_sdk(client: openai.OpenAI, model: str) -> None:
    # Streaming is not implemented yet, so Relay answers 501 in OpenAI error format.
    # This test changes when streaming lands; the point is that the SDK raises a
    # typed error carrying Relay's message, not a generic decode failure.
    with pytest.raises(openai.APIStatusError) as excinfo:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], stream=True
        )
    assert excinfo.value.status_code == 501
    assert "Streaming is not supported" in excinfo.value.message
