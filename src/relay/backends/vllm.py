"""vLLM OpenAI-compatible server (`vllm serve`)."""

from relay.backends.openai_compat import OpenAICompatBackend


class VLLMBackend(OpenAICompatBackend):
    """Observed with vLLM 0.31.0 on a Colab Tesla T4 (2026-10-10):

    - `/health` returns 200 once the engine is up (~3 min including model download).
    - Serves the name given by `--served-model-name`, so it can use the public name.
    - Rejects any other `model` with 404 (`"type": "NotFoundError"`, numeric `"code"`).
    - Supports `stream_options.include_usage` (usage chunk with `choices: []`, as llama.cpp).
    - Default port is 8000, which clashes with Relay; start it with `--port 8001`.
    - T4 (compute capability 7.5) has no bfloat16: start with `--dtype half`.
    """

    engine = "vllm"
