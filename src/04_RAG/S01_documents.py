"""Document ingestion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str]


def load_text_file(path: str | Path) -> Document:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    return Document(id=file_path.stem, text=text, metadata={"source": str(file_path)})

