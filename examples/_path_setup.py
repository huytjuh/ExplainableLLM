"""Add roadmap source folders to sys.path for standalone examples."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FOLDERS = (
    "src/01_NLP",
    "src/02_transformer",
    "src/03_LLM",
    "src/04_RAG",
    "src/05_vector_search",
    "src/06_evaluation",
    "src/07_tracing",
    "src/08_LLMOps",
)

for folder in reversed(SOURCE_FOLDERS):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

