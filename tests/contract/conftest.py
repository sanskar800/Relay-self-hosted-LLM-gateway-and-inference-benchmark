"""Contract tests talk to a running stack (Relay + backend) through the official OpenAI SDK.

Start it first:  make up  and  make run
Locally, the tests skip if Relay is not reachable. CI sets RELAY_REQUIRE_STACK=1 so a
missing stack fails instead of silently skipping.
"""

import os

import httpx
import openai
import pytest
from dotenv import dotenv_values

BASE_URL = os.environ.get("RELAY_TEST_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("RELAY_TEST_MODEL", "qwen2.5-1.5b-instruct")
# A real Relay API key: from the environment (CI), else from the git-ignored .env.
# Create one with: uv run relay-admin create-key dev --name contract-tests
API_KEY = os.environ.get("RELAY_TEST_API_KEY") or dotenv_values(".env").get("RELAY_TEST_API_KEY")


@pytest.fixture(scope="session")
def relay_up() -> None:
    health_url = BASE_URL.removesuffix("/v1") + "/healthz"
    try:
        httpx.get(health_url, timeout=2).raise_for_status()
    except httpx.HTTPError as exc:
        message = f"Relay not reachable at {health_url} ({exc!r}); run `make up` and `make run`"
        if os.environ.get("RELAY_REQUIRE_STACK") == "1":
            pytest.fail(message)
        pytest.skip(message)


@pytest.fixture
def api_key(relay_up: None) -> str:
    if not API_KEY:
        message = "no RELAY_TEST_API_KEY (env or .env); create one with relay-admin create-key"
        if os.environ.get("RELAY_REQUIRE_STACK") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return API_KEY


@pytest.fixture
def client(api_key: str) -> openai.OpenAI:
    # max_retries=0: the SDK retries 5xx and timeouts by default, which would hide
    # (and multiply) failures in a test.
    return openai.OpenAI(base_url=BASE_URL, api_key=api_key, max_retries=0, timeout=60)


@pytest.fixture
def model() -> str:
    return MODEL
