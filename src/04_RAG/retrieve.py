"""RAG retrieval and context assembly."""

from __future__ import annotations

from embeddings import HashingEmbeddingModel
from vector_store import InMemoryVectorStore, SearchResult


def retrieve(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    top_k: int = 3,
) -> list[SearchResult]:
    return vector_store.search(embedding_model.embed(question), top_k=top_k)


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        source = result.record.metadata.get("source", result.record.id)
        blocks.append(f"[{index}] source={source} score={result.score:.3f}\n{result.record.text}")
    return "\n\n".join(blocks)
