from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from S01_client import GeminiClient, LLMResponse, LLMSettings, MissingAPIKeyError


def test_llm_settings_loads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "gemini.yaml"
    config_path.write_text(
        """
        provider: gemini
        model: gemini-2.5-flash-lite
        base_url: null
        generation:
          temperature: 0.2
          max_output_tokens: 128
        """,
        encoding="utf-8",
    )

    settings = LLMSettings.from_yaml(config_path)

    assert settings.provider == "gemini"
    assert settings.model == "gemini-2.5-flash-lite"
    assert settings.base_url is None
    assert settings.generation == {
        "temperature": 0.2,
        "max_output_tokens": 128,
    }


def test_llm_settings_empty_yaml_uses_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    settings = LLMSettings.from_yaml(config_path)

    assert settings.provider == "gemini"
    assert settings.model is None
    assert settings.base_url is None
    assert settings.generation == {}


def test_gemini_client_loads_repo_config() -> None:
    client = GeminiClient(model="unused", api_key="unused", base_url="")

    settings = client._load_config()

    assert settings.provider == "gemini"
    assert settings.model == "gemini-2.5-flash-lite"
    assert "temperature" in settings.generation


def test_gemini_generate_requires_api_key() -> None:
    client = GeminiClient(model="gemini-test", api_key="", base_url="")

    with pytest.raises(MissingAPIKeyError):
        client.generate("hello")


def test_gemini_generate_uses_sdk_and_returns_llm_response() -> None:
    fake_models = FakeModels()

    class FakeGenAI:
        class Client:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.models = fake_models

    class FakeTypes:
        GenerateContentConfig = FakeGenerateContentConfig
        HttpOptions = FakeHttpOptions

    class FakeGeminiClient(GeminiClient):
        def _load_sdk(self) -> tuple[Any, Any]:
            return FakeGenAI, FakeTypes

    client = FakeGeminiClient(
        model="gemini-test",
        api_key="test-key",
        base_url="https://example.test",
    )

    response = client.generate(
        "Explain tokenization.",
        generation={"temperature": 0.2, "max_output_tokens": 16},
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "fake Gemini response"
    assert response.provider == "gemini"
    assert response.model == "gemini-test"
    assert response.latency_ms >= 0
    assert response.usage_metadata == {"total_token_count": 7}
    assert isinstance(response.raw_response, FakeResponse)

    assert fake_models.last_call is not None
    assert fake_models.last_call["model"] == "gemini-test"
    assert fake_models.last_call["contents"] == "Explain tokenization."

    config = fake_models.last_call["config"]
    assert isinstance(config, FakeGenerateContentConfig)
    assert config.kwargs == {"temperature": 0.2, "max_output_tokens": 16}


class FakeResponse:
    text = "fake Gemini response"
    usage_metadata = {"total_token_count": 7}


class FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeHttpOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeModels:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> FakeResponse:
        self.last_call = kwargs
        return FakeResponse()
