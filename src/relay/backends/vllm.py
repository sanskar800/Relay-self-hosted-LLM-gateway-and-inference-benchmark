"""vLLM OpenAI-compatible server (`vllm serve`)."""

from relay.backends.openai_compat import OpenAICompatBackend


class VLLMBackend(OpenAICompatBackend):
    """Expected behaviour, to be confirmed in the Colab check:

    - `/health` returns 200 once the engine is running.
    - Rejects requests whose `model` does not match `--served-model-name`, so the
      public model name may need mapping in the model/backend config.
    """

    engine = "vllm"
