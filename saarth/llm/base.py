"""Provider-agnostic LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # system | user | assistant | tool
    content: str
    tool_call_id: str | None = None
    name: str | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class BaseLLM(ABC):
    """Every provider implements `generate`. Tool schemas use the OpenAI JSON format."""

    name: str = "base"
    supports_tools: bool = False

    @abstractmethod
    def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...
