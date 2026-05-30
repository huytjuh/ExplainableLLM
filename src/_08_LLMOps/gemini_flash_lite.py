"""Gemini 2.5 Flash Lite API wrapper.

The wrapper imports the optional Google SDK lazily so the rest of the project
can run without network credentials or cloud dependencies.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    model: str
    latency_ms: int


class GeminiFlashLiteClient:
    def __init__(self, api_key: str | None=None, model: str | None=None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    def generate(self, prompt: str) -> GeminiResponse:
        if not self.api_key:
            raise RuntimeError("Set GEMINI_API_KEY before calling the Gemini API.")

        start = time.perf_counter()
        try:
            from google import genai  # type: ignore
        except ImportError as error:
            raise RuntimeError("Install the optional dependency with: pip install .[gemini]") from error

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model, contents=prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return GeminiResponse(text=getattr(response, "text", ""), model=self.model, latency_ms=latency_ms)

