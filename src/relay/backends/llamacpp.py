"""llama.cpp `llama-server`."""

from relay.backends.openai_compat import OpenAICompatBackend


class LlamaCppBackend(OpenAICompatBackend):
    """Observed with build b11459:

    - `/health` returns 503 while the model loads and 200 once it can serve.
    - Supports `stream_options.include_usage` (usage arrives in a chunk with `choices: []`).
    - Serves one model and does not reject a mismatched `model` name.
    - Closing the connection mid-stream cancels generation (`stop: cancel task` in logs).
    - Sometimes closes a kept-alive connection after a streamed response without telling
      the client, so the next request on it fails with "Server disconnected without
      sending a response" (docs/findings.md F7). Hence no keep-alive by default.
    """

    engine = "llamacpp"
    keep_alive_default = False
