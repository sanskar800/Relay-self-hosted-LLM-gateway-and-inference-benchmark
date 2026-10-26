"""Graceful shutdown of a real `python -m relay.serve` process under a real signal.

Needs the Compose stack (Postgres for the key, Redis, llama.cpp) and RELAY_TEST_API_KEY.
On Windows the signal is Ctrl+Break, sent to a process started in its own group; a plain
`kill` there is TerminateProcess, which no handler can see. Elsewhere it is SIGTERM.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import httpx
import pytest
from dotenv import dotenv_values

pytestmark = pytest.mark.integration

PORT = 8017
BASE = f"http://127.0.0.1:{PORT}"
API_KEY = os.environ.get("RELAY_TEST_API_KEY") or dotenv_values(".env").get("RELAY_TEST_API_KEY")
# The served model: CI runs the 0.5B model (RELAY_TEST_MODEL), development the 1.5B.
MODEL = os.environ.get("RELAY_TEST_MODEL", "qwen2.5-1.5b-instruct")


def unavailable(message: str) -> None:
    if os.environ.get("RELAY_REQUIRE_STACK") == "1":
        pytest.fail(message)
    pytest.skip(message)


def send_shutdown_signal(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.send_signal(signal.SIGTERM)


def test_signal_drains_in_flight_stream_then_exits(tmp_path) -> None:
    if not API_KEY:
        unavailable("no RELAY_TEST_API_KEY")
    log_path = tmp_path / "relay.log"
    env = {
        **os.environ,
        "RELAY_PORT": str(PORT),
        "RELAY_SHUTDOWN_GRACE_S": "2",
        "RELAY_DRAIN_TIMEOUT_S": "30",
    }
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "relay.serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    try:
        for _ in range(60):
            try:
                if httpx.get(f"{BASE}/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            unavailable(f"relay.serve did not become ready: {log_path.read_text()[-500:]}")

        headers = {"Authorization": f"Bearer {API_KEY}"}
        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Write a long story about the sea."}],
            "max_tokens": 120,
            "stream": True,
        }
        first_chunk, received = threading.Event(), bytearray()

        def stream() -> None:
            with httpx.stream(
                "POST", f"{BASE}/v1/chat/completions", json=body, headers=headers, timeout=60
            ) as resp:
                for chunk in resp.iter_bytes():
                    received.extend(chunk)
                    first_chunk.set()

        reader = threading.Thread(target=stream)
        reader.start()
        assert first_chunk.wait(30), "stream never started"

        send_shutdown_signal(proc)
        for _ in range(20):  # the signal is handled within moments
            if httpx.get(f"{BASE}/readyz", timeout=1).status_code == 503:
                break
            time.sleep(0.05)
        else:
            pytest.fail("never became unready")
        # Within the grace period: still alive and reachable, but refusing new work.
        assert httpx.get(f"{BASE}/healthz", timeout=1).status_code == 200
        refused = httpx.post(
            f"{BASE}/v1/chat/completions", json={**body, "stream": False}, headers=headers
        )
        assert refused.status_code == 503

        reader.join(60)
        assert bytes(received).rstrip().endswith(b"data: [DONE]"), "in-flight stream was cut"
        assert proc.wait(timeout=40) is not None  # exited on its own after draining
    finally:
        if proc.poll() is None:
            proc.kill()
    log_text = log_path.read_text(errors="replace")
    assert "shutdown started" in log_text
