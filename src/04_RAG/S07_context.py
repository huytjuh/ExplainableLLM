"""Prompt context assembly for RAG."""

from __future__ import annotations

from S04_indexing import SearchResult


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        source = result.record.metadata.get("source", result.record.id)
        blocks.append(f"[{index}] source={source} score={result.score:.3f}\n{result.record.text}")
    return "\n\n".join(blocks)


def build_grounded_prompt(question: str, context: str) -> str:
    return f"""Answer the question using only the retrieved context.

Question:
{question}

Retrieved context:
{context}

If the context is insufficient, say that the answer is not available in the retrieved sources."""
