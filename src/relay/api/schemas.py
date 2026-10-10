"""OpenAI chat-completions shapes: only the fields Relay relies on are validated.

Every model allows extra fields, so parameters Relay does not know about (seed, tools,
response_format, engine-specific options) reach the backend unchanged.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(_OpenAIModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    # A string, a list of content parts (text/images), or null (assistant tool calls).
    content: str | list[dict] | None = None


class StreamOptions(_OpenAIModel):
    include_usage: bool = False


class ChatCompletionRequest(_OpenAIModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    # Several choices per request complicate streaming and accounting; not supported.
    n: int | None = Field(default=None, ge=1, le=1)

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "model": "qwen2.5-1.5b-instruct",
                    "messages": [{"role": "user", "content": "What is an API gateway?"}],
                    "max_tokens": 60,
                }
            ]
        },
    )


class Usage(_OpenAIModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
