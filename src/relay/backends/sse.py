"""Server-Sent Events framing."""

from collections.abc import AsyncIterator


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Re-cut a byte stream into whole SSE events (each ending in a blank line).

    TCP chunks can split an event anywhere. Yielding whole events means a mid-stream
    failure never leaves half an event in front of the error event the gateway sends.
    Each event is yielded as soon as its terminating blank line arrives.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk.replace(b"\r\n", b"\n")
        while (end := buffer.find(b"\n\n")) != -1:
            yield buffer[: end + 2]
            buffer = buffer[end + 2 :]
    if buffer.strip():
        # Stream ended without the final blank line: still deliver the last event.
        yield buffer.rstrip(b"\n") + b"\n\n"
