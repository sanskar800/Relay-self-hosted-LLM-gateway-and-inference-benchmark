"""A scripted, in-process OpenAI-compatible engine for tests (no model, no network).

Mimics the wire behaviour observed from llama.cpp:
- non-streaming: one chat.completion with usage;
- streaming: a role chunk (no content), one chunk per token, a finish_reason chunk,
  a usage chunk with `choices: []` ONLY if `stream_options.include_usage` was requested,
  then `data: [DONE]`.

Use `FakeEngine(...).transport()` as the httpx transport under a real backend.
"""

import json
from collections.abc import AsyncIterator

import httpx

MODEL = "fake-model"


class FakeEngine:
    def __init__(self, tokens: list[str] | None = None, prompt_tokens: int = 7) -> None:
        self.tokens = tokens if tokens is not None else ["Hel", "lo", "!"]
        self.prompt_tokens = prompt_tokens
        self.requests: list[dict] = []  # parsed JSON bodies, in order

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    # --- request handling -------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path != "/v1/chat/completions" or request.method != "POST":
            return httpx.Response(404, json={"error": {"message": "not found", "type": "x"}})
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=self._events(body, include_usage),
            )
        return httpx.Response(200, json=self._completion(body))

    # --- responses ------------------------------------------------------------------------

    def _usage(self) -> dict:
        completion = len(self.tokens)
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": self.prompt_tokens + completion,
        }

    def _completion(self, body: dict) -> dict:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(self.tokens)},
                    "finish_reason": "stop",
                }
            ],
            "usage": self._usage(),
        }

    def _chunk(self, body: dict, choices: list[dict], **extra) -> bytes:
        chunk = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body.get("model", MODEL),
            "choices": choices,
            **extra,
        }
        return b"data: " + json.dumps(chunk).encode() + b"\n\n"

    async def _events(self, body: dict, include_usage: bool) -> AsyncIterator[bytes]:
        delta = {"role": "assistant", "content": None}
        yield self._chunk(body, [{"index": 0, "delta": delta, "finish_reason": None}])
        for token in self.tokens:
            yield self._chunk(
                body, [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
            )
        yield self._chunk(body, [{"index": 0, "delta": {}, "finish_reason": "stop"}])
        if include_usage:
            yield self._chunk(body, [], usage=self._usage())
        yield b"data: [DONE]\n\n"
