"""Token usage per request: read from the engine's own report, estimated only as a fallback.

Streams: engines report usage only when asked (`stream_options.include_usage`), so Relay
asks for it itself and removes the extra usage chunk if the client did not ask for it.
If no usage ever arrives (engine ignores the option, or the client disconnected first),
completion tokens are estimated as the number of content chunks and labelled as such.
"""

import json
from dataclasses import dataclass
from typing import Literal

UsageSource = Literal["engine", "estimated", "missing"]


@dataclass(frozen=True)
class UsageRecord:
    model: str
    backend: str
    stream: bool
    status_code: int
    prompt_tokens: int | None
    completion_tokens: int | None
    # "engine": reported by the engine (exact). "estimated": counted by Relay from
    # streamed chunks (approximate; prompt unknown). "missing": nothing to go on.
    source: UsageSource

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


def usage_from_completion(content: bytes) -> tuple[int | None, int | None, UsageSource]:
    """Usage from a non-streaming chat.completion body."""
    try:
        usage = json.loads(content).get("usage") or {}
    except (ValueError, AttributeError):
        return None, None, "missing"
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt, completion, "engine"
    return None, None, "missing"


def request_stream_usage(body: bytes) -> bytes:
    """Ask the engine to report usage at the end of the stream.

    The only change Relay ever makes to a client's request. Every other field is kept;
    the JSON is re-serialised, so whitespace may differ from the client's bytes.
    """
    data = json.loads(body)
    options = data.get("stream_options") or {}
    data["stream_options"] = {**options, "include_usage": True}
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


class StreamUsage:
    """Watches a stream's SSE events; decides which to forward; remembers the usage."""

    def __init__(self, *, forward_usage_chunk: bool) -> None:
        self._forward_usage_chunk = forward_usage_chunk
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.content_chunks = 0

    def observe(self, event: bytes) -> bool:
        """Record what the event says; return False if it must not reach the client."""
        payload = _data_payload(event)
        if payload is None:
            return True
        try:
            chunk = json.loads(payload)
        except ValueError:
            return True
        if not isinstance(chunk, dict):
            return True

        choices = chunk.get("choices") or []
        if any((c.get("delta") or {}).get("content") for c in choices if isinstance(c, dict)):
            self.content_chunks += 1

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")
            if not choices and not self._forward_usage_chunk:
                return False  # the usage-only chunk Relay asked for, not the client
        return True

    def result(self) -> tuple[int | None, int | None, UsageSource]:
        if isinstance(self.prompt_tokens, int) and isinstance(self.completion_tokens, int):
            return self.prompt_tokens, self.completion_tokens, "engine"
        if self.content_chunks:
            return None, self.content_chunks, "estimated"
        return None, None, "missing"


def _data_payload(event: bytes) -> bytes | None:
    """The JSON after `data:` in an SSE event, or None (comments, [DONE], other fields)."""
    for line in event.split(b"\n"):
        if line.startswith(b"data:"):
            payload = line[5:].strip()
            return payload if payload.startswith(b"{") else None
    return None
