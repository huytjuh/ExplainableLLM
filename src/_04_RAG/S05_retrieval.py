"""Retrieve relevant chunks for a user question."""

from __future__ import annotations

from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore, SearchResult


def retrieve(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    top_k: int=3,
) -> list[SearchResult]:
    return vector_store.search(embedding_model.embed(question), top_k=top_k)
