"""Inference backends behind one interface."""

from relay.backends.base import Backend
from relay.backends.llamacpp import LlamaCppBackend
from relay.backends.openai_compat import OpenAICompatBackend
from relay.backends.vllm import VLLMBackend

# Engine name (as written in config) -> implementation.
ENGINES: dict[str, type[OpenAICompatBackend]] = {
    "llamacpp": LlamaCppBackend,
    "vllm": VLLMBackend,
}

__all__ = ["ENGINES", "Backend", "LlamaCppBackend", "OpenAICompatBackend", "VLLMBackend"]
