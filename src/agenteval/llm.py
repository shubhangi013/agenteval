from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from agenteval.config import Settings


@dataclass(frozen=True)
class LLMResponse:
    text: str
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        if json_mode and "JUDGE" in prompt:
            return LLMResponse('{"score": 4, "reason": "mock deterministic judge"}', 0.0)
        return LLMResponse(prompt, 0.0)


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        input_rate: float,
        output_rate: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.input_rate = input_rate
        self.output_rate = output_rate

    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return LLMResponse(
            text=str(data["choices"][0]["message"]["content"]),
            cost_usd=(
                input_tokens * self.input_rate + output_tokens * self.output_rate
            )
            / 1_000_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class AzureOpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        deployment: str,
        endpoint: str,
        api_version: str,
        input_rate: float,
        output_rate: float,
    ) -> None:
        if not api_key:
            raise ValueError("AGENTEVAL_AZURE_OPENAI_API_KEY is required")
        if not deployment:
            raise ValueError("AGENTEVAL_AZURE_OPENAI_DEPLOYMENT is required")
        if not endpoint:
            raise ValueError("AGENTEVAL_AZURE_OPENAI_ENDPOINT is required")
        self.api_key = api_key
        self.deployment = deployment
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.input_rate = input_rate
        self.output_rate = output_rate

    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        deployment = quote(self.deployment, safe="")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/openai/deployments/{deployment}/chat/completions",
                headers={"api-key": self.api_key, "content-type": "application/json"},
                params={"api-version": self.api_version},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        return LLMResponse(
            text=str(data["choices"][0]["message"]["content"]),
            cost_usd=(
                input_tokens * self.input_rate + output_tokens * self.output_rate
            )
            / 1_000_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=str(data["response"]),
            cost_usd=0.0,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        input_rate: float,
        output_rate: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.input_rate = input_rate
        self.output_rate = output_rate

    async def complete(self, prompt: str, *, json_mode: bool = False) -> LLMResponse:
        suffix = "\nReturn only valid JSON." if json_mode else ""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt + suffix}],
                },
                timeout=60,
            )
            response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        return LLMResponse(
            text=str(data["content"][0]["text"]),
            cost_usd=(
                input_tokens * self.input_rate + output_tokens * self.output_rate
            )
            / 1_000_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def create_llm_client(provider: str, settings: Settings) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(
            settings.openai_api_key,
            settings.openai_model,
            settings.openai_base_url,
            settings.openai_input_usd_per_million,
            settings.openai_output_usd_per_million,
        )
    if provider == "azure_openai":
        return AzureOpenAIClient(
            settings.azure_openai_api_key,
            settings.azure_openai_deployment,
            settings.azure_openai_endpoint,
            settings.azure_openai_api_version,
            settings.azure_openai_input_usd_per_million,
            settings.azure_openai_output_usd_per_million,
        )
    if provider == "anthropic":
        return AnthropicClient(
            settings.anthropic_api_key,
            settings.anthropic_model,
            settings.anthropic_base_url,
            settings.anthropic_input_usd_per_million,
            settings.anthropic_output_usd_per_million,
        )
    if provider == "ollama":
        return OllamaClient(settings.ollama_model, settings.ollama_base_url)
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"unknown LLM provider: {provider}")


def parse_json_response(response: LLMResponse) -> dict[str, Any]:
    parsed = json.loads(response.text)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed
