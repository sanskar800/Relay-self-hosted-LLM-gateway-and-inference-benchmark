"""Connection reuse per engine, over real sockets (docs/findings.md F7).

llama.cpp sometimes drops a kept-alive connection after a streamed response, so Relay opens
a new connection per request to llama.cpp backends; other engines keep pooling.
"""

import asyncio
import json

import pytest

from relay.backends import LlamaCppBackend, VLLMBackend
from relay.config import RelayConfig
from tests.engine_server import serve_engine
from tests.fake_engine import FakeEngine

pytestmark = pytest.mark.integration

BODY = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()


async def three_requests(backend) -> None:
    try:
        for _ in range(3):
            await backend.chat(BODY)
            stream = await backend.stream(BODY)
            async for _ in stream.events():
                pass
    finally:
        await backend.aclose()


def ports_used(backend_class, url: str, engine: FakeEngine, **kw) -> list[int]:
    asyncio.run(three_requests(backend_class("b", url, **kw)))
    return engine.client_ports


def test_llamacpp_opens_a_new_connection_per_request() -> None:
    engine = FakeEngine()
    with serve_engine(engine) as url:
        ports = ports_used(LlamaCppBackend, url, engine)
    assert len(ports) == 6 and len(set(ports)) == 6


def test_other_engines_reuse_connections() -> None:
    engine = FakeEngine()
    with serve_engine(engine) as url:
        ports = ports_used(VLLMBackend, url, engine)
    assert len(ports) == 6 and len(set(ports)) == 1


def test_config_can_override_the_engine_default() -> None:
    engine = FakeEngine()
    with serve_engine(engine) as url:
        ports = ports_used(LlamaCppBackend, url, engine, keep_alive=True)
    assert len(set(ports)) == 1
    config = RelayConfig.model_validate(
        {
            "backends": {"b": {"engine": "llamacpp", "url": "http://x", "keep_alive": True}},
            "models": {"m": {"backends": ["b"]}},
        }
    )
    assert config.backends["b"].keep_alive is True
