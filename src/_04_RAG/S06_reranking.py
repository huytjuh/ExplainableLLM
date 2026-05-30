"""Reranking helpers for retrieved RAG chunks."""

from __future__ import annotations

from S04_indexing import SearchResult


def rerank_by_score(results: list[SearchResult]) -> list[SearchResult]:
    """Return retrieved results in descending relevance-score order."""

    return sorted(results, key=lambda result: result.score, reverse=True)


def keep_above_score(results: list[SearchResult], *, minimum_score: float) -> list[SearchResult]:
    """Filter out weak retrieval matches before context assembly."""

    return [result for result in results if result.score >= minimum_score]
