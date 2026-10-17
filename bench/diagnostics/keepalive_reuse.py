"""Does an engine drop reused (kept-alive) connections? Findings F7.

Sends 40 realistic streamed chat requests one after another over ONE reused connection
and counts failures. Standalone (only httpx), so it can run inside the Docker network to
take Windows port forwarding out of the path:

    uv run python bench/diagnostics/keepalive_reuse.py http://127.0.0.1:8082/v1/chat/completions
    docker run --rm --network relay_default -v "$PWD/bench/diagnostics:/w:ro" python:3.12-slim \\
        sh -c "pip install -q httpx && python /w/keepalive_reuse.py http://llamacpp-gpu:8080/v1/chat/completions"

Add --fresh to open a new connection for every request (the control).
"""

import random
import sys

import httpx

URL = sys.argv[1]
FRESH = "--fresh" in sys.argv
N = 40
WORDS = [
    "river",
    "mountain",
    "city",
    "library",
    "garden",
    "market",
    "bridge",
    "harbor",
    "forest",
    "valley",
]

rng = random.Random()
fails = 0
limits = httpx.Limits(max_connections=1, max_keepalive_connections=0 if FRESH else 1)
with httpx.Client(limits=limits, timeout=60) as client:
    for i in range(N):
        prompt = f"[note {rng.getrandbits(48):012x}] Continue: " + " ".join(
            rng.choice(WORDS) for _ in range(int(rng.lognormvariate(5.0, 0.6)))
        )
        body = {
            "model": "qwen2.5-1.5b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(min(256, max(16, rng.lognormvariate(4.56, 0.5)))),
            "stream": True,
            "ignore_eos": True,
            "temperature": 0,
        }
        try:
            with client.stream("POST", URL, json=body) as r:
                for _ in r.iter_lines():
                    pass
        except httpx.HTTPError as e:
            fails += 1
            print(f"request {i}: {type(e).__name__}: {e}")
mode = "new connection each time" if FRESH else "one reused connection"
print(f"{mode}: {N - fails}/{N} ok")
