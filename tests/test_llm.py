from typing import Any, Self

import pytest

from agenteval.llm import AzureOpenAIClient


@pytest.mark.asyncio
async def test_azure_openai_uses_deployment_route_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": '{"answer": 4}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    class FakeClient:
        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def post(self, url: str, **kwargs: Any) -> FakeResponse:
            captured["url"] = url
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr("agenteval.llm.httpx.AsyncClient", FakeClient)
    client = AzureOpenAIClient(
        api_key="secret",
        deployment="gpt-5.6-sol",
        endpoint="https://example.services.ai.azure.com/",
        api_version="2024-10-21",
        input_rate=1.0,
        output_rate=2.0,
    )

    response = await client.complete("Return JSON", json_mode=True)

    assert captured["url"] == (
        "https://example.services.ai.azure.com/openai/deployments/"
        "gpt-5.6-sol/chat/completions"
    )
    assert captured["headers"] == {
        "api-key": "secret",
        "content-type": "application/json",
    }
    assert captured["params"] == {"api-version": "2024-10-21"}
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert "temperature" not in captured["json"]
    assert response.text == '{"answer": 4}'
    assert response.cost_usd == 0.00002