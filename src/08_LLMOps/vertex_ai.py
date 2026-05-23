"""Vertex AI configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VertexAIConfig:
    project_id: str
    location: str = "us-central1"

    @classmethod
    def from_env(cls) -> "VertexAIConfig":
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise RuntimeError("Set GOOGLE_CLOUD_PROJECT for Vertex AI examples.")
        return cls(project_id=project_id, location=os.getenv("VERTEX_AI_LOCATION", "us-central1"))

    @property
    def resource_prefix(self) -> str:
        return f"projects/{self.project_id}/locations/{self.location}"

