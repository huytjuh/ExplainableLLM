"""Azure DevOps artifact metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AzurePipelineArtifact:
    name: str
    path: str

    def publish_yaml(self) -> str:
        normalized = self.path.replace("\\", "/")
        return (
            "- task: PublishPipelineArtifact@1\n"
            "  inputs:\n"
            f"    targetPath: '{normalized}'\n"
            f"    artifact: '{self.name}'\n"
            "    publishLocation: 'pipeline'\n"
        )


def discover_artifacts(root: str | Path = "artifacts") -> list[AzurePipelineArtifact]:
    base = Path(root)
    return [
        AzurePipelineArtifact(name=child.name, path=str(child))
        for child in base.iterdir()
        if child.is_dir()
    ] if base.exists() else []

