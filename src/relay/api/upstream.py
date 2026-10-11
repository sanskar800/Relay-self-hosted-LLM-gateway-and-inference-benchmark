"""The request body Relay sends to a backend.

Relay forwards the client's bytes unchanged, except for two documented additions:
- `stream_options.include_usage` on streams, so every request can be accounted for;
- `max_tokens` = the tenant's cap when the client sent none, so the cap cannot be
  bypassed by omitting the field.
When either applies, the JSON is re-serialised once: every field is kept, but
whitespace may differ from the client's bytes.
"""

import json


def upstream_body(body: bytes, *, include_usage: bool, default_max_tokens: int | None) -> bytes:
    if not include_usage and default_max_tokens is None:
        return body
    data = json.loads(body)
    if include_usage:
        options = data.get("stream_options") or {}
        data["stream_options"] = {**options, "include_usage": True}
    if default_max_tokens is not None:
        data["max_tokens"] = default_max_tokens
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
