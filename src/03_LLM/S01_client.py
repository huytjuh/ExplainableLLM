"""Universal LLM client with YAML-backed provider configs."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

class LLMClientError(RuntimeError):
    """Base error for direct LLM client failures."""

class MissingAPIKeyError(LLMClientError):
    """Raised when a provider API key is not configured."""

class MissingSDKError(LLMClientError):
    """Raised when an optional provider SDK is not installed."""

class UnknownProviderError(LLMClientError):
    """Raised when an unsupported provider is requested."""

@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str | None=None
    base_url: str | None=None
    generation: dict[str, Any]=field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LLMSettings":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            provider=data.get("provider", "gemini"),
            model=data.get("model"),
            base_url=data.get("base_url"),
            generation=data.get("generation") or {},
        )


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    raw_response: Any | None=field(default=None, repr=False)
    usage_metadata: Any | None=field(default=None, repr=False)


class LLMClient(ABC):
    """Base class for LLM clients."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def generate(self, prompt: str, generation: dict[str, Any] | None=None) -> LLMResponse:
        raise NotImplementedError
    
    @abstractmethod
    def _load_sdk(self) -> tuple[Any, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def _load_config(self) -> LLMSettings:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Gemini LLM client."""

    def generate(self, prompt: str, generation: dict[str, Any] | None=None) -> LLMResponse:
        """Generate text from a prompt."""
        if not self.api_key:
            raise MissingAPIKeyError("Set GEMINI_API_KEY before calling the Gemini API.")

        genai, types = self._load_sdk()

        client_kwargs: dict[str, Any]={"api_key": self.api_key}
        if self.base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=self.base_url)           # OPTIONAL: Custom base url gemini compatible endpoint

        client = genai.Client(**client_kwargs)
        config = types.GenerateContentConfig(**generation) if generation else None

        started_at = time.perf_counter()
        response = client.models.generate_content(model=self.model, contents=prompt, config=config)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        return LLMResponse(
            text=getattr(response, "text", "") or "",
            provider="gemini",
            model=self.model,
            latency_ms=latency_ms,
            raw_response=response,
            usage_metadata=getattr(response, "usage_metadata", None),
        )

    def _load_sdk(self) -> tuple[Any, Any]:
        """Load the Gemini SDK."""
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as error:
            raise MissingSDKError("Install Gemini support with: poetry install --extras gemini") from error
        return genai, types

    def _load_config(self) -> LLMSettings:
        """Load the Gemini config."""
        config_path = Path("config/gemini.yaml")
        if not config_path.is_absolute() and not config_path.exists():
            config_path = Path(__file__).resolve().parents[2] / config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Gemini config file not found: {config_path}")

        return LLMSettings.from_yaml(config_path)
