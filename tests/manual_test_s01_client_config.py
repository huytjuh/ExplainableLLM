"""Manual smoke test for src/03_LLM/S01_client.py config loading.

Run with:
    poetry run python tests/manual_test_s01_client_config.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_SRC = REPO_ROOT / "src" / "03_LLM"
sys.path.insert(0, str(LLM_SRC))

from S01_client import GeminiClient, LLMResponse, LLMSettings, MissingAPIKeyError  # noqa: E402


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def test_settings_from_yaml() -> None:
    settings = LLMSettings.from_yaml(REPO_ROOT / "config" / "gemini.yaml")

    assert_equal(settings.provider, "gemini", "provider")
    assert_equal(settings.model, "gemini-2.5-flash-lite", "model")
    assert "temperature" in settings.generation
    assert "max_output_tokens" in settings.generation


def test_settings_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "empty.yaml"
        path.write_text("", encoding="utf-8")

        settings = LLMSettings.from_yaml(path)

    assert_equal(settings.provider, "gemini", "default provider")
    assert_equal(settings.model, None, "default model")
    assert_equal(settings.base_url, None, "default base_url")
    assert_equal(settings.generation, {}, "default generation")


def test_gemini_client_load_config() -> None:
    client = GeminiClient(model="unused", api_key="unused", base_url="")
    settings = client._load_config()

    assert_equal(settings.provider, "gemini", "client config provider")
    assert_equal(settings.model, "gemini-2.5-flash-lite", "client config model")
    assert "temperature" in settings.generation


def test_gemini_client_generate_requires_api_key() -> None:
    client = GeminiClient(model="gemini-test", api_key="", base_url="")

    try:
        client.generate("hello", generation={})
    except MissingAPIKeyError:
        return

    raise AssertionError("generate should raise MissingAPIKeyError when api_key is empty")


def test_gemini_client_generate_wraps_sdk_response() -> None:
    class FakeResponse:
        text = "hello from fake gemini"
        usage_metadata = {"prompt_token_count": 3}

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeHttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeModels:
        def __init__(self) -> None:
            self.last_call: dict[str, object] | None = None

        def generate_content(self, **kwargs: object) -> FakeResponse:
            self.last_call = kwargs
            return FakeResponse()

    class FakeGenAIClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.models = fake_models

    class FakeGenAI:
        Client = FakeGenAIClient

    class FakeTypes:
        GenerateContentConfig = FakeConfig
        HttpOptions = FakeHttpOptions

    class FakeGeminiClient(GeminiClient):
        def _load_sdk(self) -> tuple[object, object]:
            return FakeGenAI, FakeTypes

    fake_models = FakeModels()
    client = FakeGeminiClient(
        model="gemini-test",
        api_key="test-key",
        base_url="https://example.test",
    )
    response = client.generate(
        "hello",
        generation={"temperature": 0.2, "max_output_tokens": 16},
    )

    assert isinstance(response, LLMResponse)
    assert_equal(response.text, "hello from fake gemini", "response text")
    assert_equal(response.provider, "gemini", "response provider")
    assert_equal(response.model, "gemini-test", "response model")
    assert_equal(response.raw_response.__class__.__name__, "FakeResponse", "raw response")
    assert_equal(response.usage_metadata, {"prompt_token_count": 3}, "usage metadata")

    if fake_models.last_call is None:
        raise AssertionError("generate_content was not called")

    assert_equal(fake_models.last_call["model"], "gemini-test", "sdk model")
    assert_equal(fake_models.last_call["contents"], "hello", "sdk contents")
    config = fake_models.last_call["config"]
    assert isinstance(config, FakeConfig)
    assert_equal(config.kwargs["temperature"], 0.2, "sdk temperature")
    assert_equal(config.kwargs["max_output_tokens"], 16, "sdk max output tokens")
    if response.latency_ms < 0:
        raise AssertionError("latency_ms should not be negative")


if __name__ == "__main__":
    test_settings_from_yaml()
    test_settings_defaults()
    test_gemini_client_load_config()
    test_gemini_client_generate_requires_api_key()
    test_gemini_client_generate_wraps_sdk_response()
    print("S01_client config smoke test passed.")
