"""Concrete LLM providers: OpenAI, Azure OpenAI, AWS Bedrock and an offline extractive model."""

from __future__ import annotations

import json
import re
from typing import Any

from saarth.config import Settings, get_settings
from saarth.llm.base import BaseLLM, LLMResponse, Message, ToolCall
from saarth.text import sentences, tokens


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.role == "tool":
            d["tool_call_id"] = m.tool_call_id
        out.append(d)
    return out


def _parse_openai(resp: Any) -> LLMResponse:
    msg = resp.choices[0].message
    calls = []
    for tc in msg.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return LLMResponse(content=msg.content or "", tool_calls=calls)


class OpenAILLM(BaseLLM):
    name = "openai"
    supports_tools = True

    
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client: Any = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model
        self.temperature = settings.temperature


    def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self.model, "messages": _to_openai_messages(messages),
                                  "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        return _parse_openai(self.client.chat.completions.create(**kwargs))


class AzureOpenAILLM(OpenAILLM):
    name = "azure"

    def __init__(self, settings: Settings) -> None:  # parent init intentionally not called
        from openai import AzureOpenAI

        if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set")
        self.client = AzureOpenAI(azure_endpoint=settings.azure_openai_endpoint,
                                  api_key=settings.azure_openai_api_key,
                                  api_version=settings.azure_openai_api_version)
        self.model = settings.azure_openai_deployment
        self.temperature = settings.temperature


class BedrockLLM(BaseLLM):
    """AWS Bedrock via the Converse API (Claude, Llama, Mistral, Titan)."""

    name = "bedrock"
    supports_tools = True

    def __init__(self, settings: Settings) -> None:
        import boto3

        self.client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        self.model_id = settings.bedrock_model_id
        self.temperature = settings.temperature

    def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        system = [{"text": m.content} for m in messages if m.role == "system"]
        convo: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"toolResult": {"toolUseId": m.tool_call_id, "content": [{"text": m.content}]}}
                convo.append({"role": "user", "content": [block]})
            else:
                convo.append({"role": m.role, "content": [{"text": m.content or " "}]})
        kwargs: dict[str, Any] = {"modelId": self.model_id, "messages": convo,
                                  "inferenceConfig": {"temperature": self.temperature}}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": [
                {"toolSpec": {"name": t["function"]["name"], "description": t["function"]["description"],
                              "inputSchema": {"json": t["function"]["parameters"]}}} for t in tools]}
        resp = self.client.converse(**kwargs)
        text, calls = "", []
        for block in resp["output"]["message"]["content"]:
            if "text" in block:
                text += block["text"]
            if "toolUse" in block:
                tu = block["toolUse"]
                calls.append(ToolCall(id=tu["toolUseId"], name=tu["name"], arguments=tu.get("input", {})))
        return LLMResponse(content=text, tool_calls=calls)


class OfflineLLM(BaseLLM):
    """Deterministic, key-free model for local demos and tests.

    It answers by extracting the context sentences that best overlap the question and keeps
    their citation markers, so the whole pipeline runs without any API key.
    """

    name = "offline"
    supports_tools = False

    def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question = re.search(r"Question:\s*(.*)", user, re.S)
        q_text = question.group(1).split("\n")[0] if question else user
        q_tokens = set(tokens(q_text))

        context_match = re.search(r"Context:\s*(.*?)\nTool results:", user, re.S)
        tools_match = re.search(r"Tool results:\s*(.*?)\nQuestion:", user, re.S)
        parts: list[str] = []
        if tools_match and tools_match.group(1).strip() not in ("", "None"):
            parts.append(tools_match.group(1).strip())

        if context_match:
            scored: list[tuple[float, str]] = []
            for block in re.split(r"\n(?=\[\d+\])", context_match.group(1).strip()):
                m = re.match(r"\[(\d+)\][^\n]*\n?(.*)", block, re.S)
                if not m:
                    continue
                cite, body = m.group(1), m.group(2)
                for s in sentences(body):
                    overlap = len(q_tokens & set(tokens(s)))
                    if overlap >= max(1, min(2, len(q_tokens) // 3)):
                        scored.append((overlap / (len(tokens(s)) ** 0.3 + 1), f"{s} [{cite}]"))
            
            scored.sort(key=lambda x: -x[0])

            # Keep only the best-matching sentence to avoid unrelated facts.
            if scored:
                parts.append(scored[0][1])


        if not parts:
            return LLMResponse(content="I don't have enough information in the knowledge base to answer that.")
        return LLMResponse(content=" ".join(parts))


def get_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAILLM(settings)
    if settings.llm_provider == "azure":
        return AzureOpenAILLM(settings)
    if settings.llm_provider == "bedrock":
        return BedrockLLM(settings)
    return OfflineLLM()
