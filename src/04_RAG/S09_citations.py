"""Citation formatting helpers for grounded answers."""

from __future__ import annotations

from S04_indexing import SearchResult


def citation_list(results: list[SearchResult]) -> list[str]:
    citations = []
    for index, result in enumerate(results, start=1):
        source = result.record.metadata.get("source", result.record.id)
        citations.append(f"[{index}] {source}")
    return citations


def append_citations(answer: str, results: list[SearchResult]) -> str:
    citations = citation_list(results)
    if not citations:
        return answer
    return f"{answer}\n\nSources:\n" + "\n".join(citations)
