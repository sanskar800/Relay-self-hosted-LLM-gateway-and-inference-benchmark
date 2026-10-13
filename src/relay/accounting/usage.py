"""Token usage per request: read from the engine's own report, estimated only as a fallback.

Streams: engines report usage only when asked (`stream_options.include_usage`), so Relay
asks for it itself (see api/upstream.py) and removes the extra usage chunk if the client
did not ask for it.
If no usage ever arrives (engine ignores the option, or the client disconnected first),
completion tokens are estimated as the number of content chunks and labelled as such.
"""

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class UsageEvent:
    """One finished request, as written to the usage ledger (table usage_events)."""

    request_id: uuid.UUID
    occurred_at: datetime
    tenant_id: str
    api_key_id: str
    usage: UsageRecord
    ttft_ms: float | None  # streams only: time to the first content token
    duration_ms: float
    cost_gbp: float | None  # None until pricing rates are known


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


class StreamUsage:
    """Watches a stream's SSE events; decides which to forward; remembers the usage."""

    def __init__(self, *, forward_usage_chunk: bool) -> None:
        self._forward_usage_chunk = forward_usage_chunk
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.content_chunks = 0
        # time.perf_counter() when the first chunk WITH content arrived (the first
        # chunk usually only carries the role, so it does not count for TTFT).
        self.first_content_at: float | None = None
        # Seconds between consecutive content chunks (inter-token latency samples).
        self.token_gaps: list[float] = []
        self._last_content_at: float | None = None

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
            now = time.perf_counter()
            if self.first_content_at is None:
                self.first_content_at = now
            if self._last_content_at is not None:
                self.token_gaps.append(now - self._last_content_at)
            self._last_content_at = now
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
