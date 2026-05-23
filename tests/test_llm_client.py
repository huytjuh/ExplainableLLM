import pytest

from S01_client import (
    BaseLLMClient,
    DEFAULT_GEMINI_MODEL,
    ClaudeClient,
    GeminiClient,
    GeminiFlashLiteClient,
    LLMClient,
    LLMSettings,
    MissingAPIKeyError,
    OllamaClient,
    OpenAIClient,
    UnknownProviderError,
    claude_kwargs,
    gemini_kwargs,
    make_provider_client,
    ollama_options,
    openai_kwargs,
    normalize_provider,
)


def test_yaml_generation_builds_provider_kwargs():
    generation = {
        "temperature": 0.1,
        "max_output_tokens": 128,
        "top_p": 0.9,
        "top_k": 20,
        "stop_sequences": ["END"],
        "response_mime_type": "application/json",
        "system_instruction": "You are concise.",
    }

    assert gemini_kwargs(generation)["top_k"] == 20
    assert gemini_kwargs(generation)["response_mime_type"] == "application/json"
    assert openai_kwargs(generation)["response_format"] == {"type": "json_object"}
    assert claude_kwargs(generation)["system"] == "You are concise."
    assert ollama_options(generation)["num_predict"] == 128


def test_provider_aliases():
    assert normalize_provider("chatgpt") == "openai"
    assert normalize_provider("anthropic") == "claude"
    assert normalize_provider("llama") == "ollama"


def test_factory_creates_provider_clients():
    assert isinstance(make_provider_client("gemini", api_key="key"), GeminiClient)
    assert isinstance(make_provider_client("chatgpt", api_key="key"), OpenAIClient)
    assert isinstance(make_provider_client("claude", api_key="key"), ClaudeClient)
    assert isinstance(make_provider_client("llamas"), OllamaClient)
    assert isinstance(make_provider_client("gemini", api_key="key"), BaseLLMClient)


def test_factory_rejects_unknown_provider():
    with pytest.raises(UnknownProviderError):
        make_provider_client("not-real")


def test_universal_client_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = GeminiClient(api_key="test-key")

    assert client.model == DEFAULT_GEMINI_MODEL


def test_settings_load_from_yaml(tmp_path):
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        """
        provider: ollama
        model: llama3.1
        base_url: http://localhost:11434
        generation:
          temperature: 0.4
          max_output_tokens: 256
          system_instruction: Be concise.
        """,
        encoding="utf-8",
    )

    settings = LLMSettings.from_yaml(config_path)
    client = make_provider_client(settings.provider, model=settings.model, base_url=settings.base_url)

    assert settings.provider == "ollama"
    assert settings.generation["temperature"] == 0.4
    assert settings.generation["max_output_tokens"] == 256
    assert client.model == "llama3.1"


def test_committed_provider_yaml_files_load():
    for path in ("config/gemini.yaml", "config/chatgpt.yaml", "config/claude.yaml", "config/ollama.yaml"):
        settings = LLMSettings.from_yaml(path)
        assert settings.provider
        assert settings.model
        assert "temperature" in settings.generation


def test_client_reads_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "custom-model")

    client = GeminiFlashLiteClient.from_env()

    assert client.api_key == "env-key"
    assert client.model == "custom-model"


def test_generate_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiFlashLiteClient()

    with pytest.raises(MissingAPIKeyError):
        client.generate("hello")
