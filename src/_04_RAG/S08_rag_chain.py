"""End-to-end RAG chain helpers."""

from __future__ import annotations

from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore, SearchResult
from S05_retrieval import retrieve
from S06_reranking import rerank_by_score
from S07_context import build_context, build_grounded_prompt


def build_rag_prompt(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    top_k: int=3,
) -> tuple[str, list[SearchResult]]:
    """Retrieve, rerank, assemble context, and return a grounded prompt."""

    results = rerank_by_score(retrieve(question, embedding_model, vector_store, top_k=top_k))
    context = build_context(results)
    return build_grounded_prompt(question, context), results
