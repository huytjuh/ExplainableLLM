"""A small in-memory vector store with cosine search and metadata filters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    record: VectorRecord
    score: float


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def add(self, record_id: str, text: str, vector: list[float], metadata: dict[str, str] | None = None) -> None:
        self._records.append(VectorRecord(record_id, text, vector, metadata or {}))

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 3,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for record in self._records:
            if metadata_filter and any(record.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            results.append(SearchResult(record=record, score=cosine_similarity(query_vector, record.vector)))
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))

