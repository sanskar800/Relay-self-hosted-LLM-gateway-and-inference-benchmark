"""How many tokens to reserve before a request, when the real count is not known yet."""

from relay.api.schemas import ChatCompletionRequest

CHARS_PER_TOKEN = 4  # rough rule for English text; settled against the real count later
PER_MESSAGE_OVERHEAD = 4  # role markers and separators added by the chat template


def estimate_prompt_tokens(payload: ChatCompletionRequest) -> int:
    chars = 0
    for message in payload.messages:
        if isinstance(message.content, str):
            chars += len(message.content)
        elif isinstance(message.content, list):
            chars += sum(len(str(part.get("text", ""))) for part in message.content)
    return chars // CHARS_PER_TOKEN + PER_MESSAGE_OVERHEAD * len(payload.messages)


def reservation(payload: ChatCompletionRequest, max_tokens_default: int) -> int:
    """Estimated prompt + the most the completion can cost."""
    completion = payload.max_tokens or payload.max_completion_tokens or max_tokens_default
    return estimate_prompt_tokens(payload) + completion
