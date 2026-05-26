"""Query translation helpers for advanced RAG retrieval.

Query translation happens before retrieval. It turns one user question into a
better retrieval query, several query variants, smaller sub-questions, or a
broader step-back question. The translated queries are then sent to the normal
retriever and merged before context assembly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore, SearchResult
from S05_retrieval import retrieve


QueryTranslator = Callable[[str], str]
MultiQueryTranslator = Callable[[str], list[str]]


@dataclass(frozen=True)
class QueryTranslationPrompts:
    """Prompt templates for LLM-backed query translation."""

    rewrite: str
    multi_query: str
    decomposition: str
    step_back: str


DEFAULT_QUERY_TRANSLATION_PROMPTS = QueryTranslationPrompts(
    rewrite=(
        "Rewrite the user question as one standalone search query for retrieval. "
        "Preserve the original meaning and do not answer the question.\n\n"
        "Question: {question}"
    ),
    multi_query=(
        "Generate {count} different search queries for retrieving evidence that "
        "can answer the user question. Return one query per line and do not "
        "answer the question.\n\nQuestion: {question}"
    ),
    decomposition=(
        "Break the user question into focused sub-questions for retrieval. "
        "Return one sub-question per line and do not answer them.\n\n"
        "Question: {question}"
    ),
    step_back=(
        "Write one broader, more general question that would help retrieve "
        "background concepts needed to answer the user question. Do not answer "
        "the question.\n\nQuestion: {question}"
    ),
)


def build_rewrite_prompt(question: str) -> str:
    """Build a prompt for rewriting a question into a standalone query."""

    return DEFAULT_QUERY_TRANSLATION_PROMPTS.rewrite.format(question=question)


def build_multi_query_prompt(question: str, *, count: int = 3) -> str:
    """Build a prompt for generating multiple retrieval queries."""

    return DEFAULT_QUERY_TRANSLATION_PROMPTS.multi_query.format(question=question, count=count)


def build_decomposition_prompt(question: str) -> str:
    """Build a prompt for decomposing a complex question into sub-questions."""

    return DEFAULT_QUERY_TRANSLATION_PROMPTS.decomposition.format(question=question)


def build_step_back_prompt(question: str) -> str:
    """Build a prompt for generating a broader step-back retrieval query."""

    return DEFAULT_QUERY_TRANSLATION_PROMPTS.step_back.format(question=question)


def parse_line_separated_queries(text: str) -> list[str]:
    """Parse LLM output into clean query strings."""

    queries: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. )").strip()
        if cleaned:
            queries.append(cleaned)
    return deduplicate_queries(queries)


def deduplicate_queries(queries: Sequence[str]) -> list[str]:
    """Keep query order while removing case-insensitive duplicates."""

    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        normalized = " ".join(query.lower().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(query.strip())
    return unique


def simple_rewrite_query(question: str) -> str:
    """Deterministic baseline rewrite for examples without an LLM."""

    return " ".join(question.strip().rstrip("?").split())


def simple_multi_queries(question: str) -> list[str]:
    """Generate small query variants without an LLM.

    This is intentionally simple for learning and tests. In production, replace
    it with an LLM-backed translator using ``build_multi_query_prompt``.
    """

    rewritten = simple_rewrite_query(question)
    return deduplicate_queries(
        [
            rewritten,
            f"definition and explanation of {rewritten}",
            f"examples and evidence for {rewritten}",
        ]
    )


def simple_decompose_query(question: str) -> list[str]:
    """Split obvious compound questions into smaller retrieval queries."""

    rewritten = simple_rewrite_query(question)
    separators = [" and ", ";", " plus "]
    parts = [rewritten]
    for separator in separators:
        next_parts: list[str] = []
        for part in parts:
            next_parts.extend(chunk.strip() for chunk in part.split(separator))
        parts = next_parts
    return deduplicate_queries(part for part in parts if part)


def simple_step_back_query(question: str) -> str:
    """Create a broad retrieval query for background context."""

    rewritten = simple_rewrite_query(question)
    return f"general principles and background for {rewritten}"


def reciprocal_rank_fusion(
    ranked_result_sets: Sequence[Sequence[SearchResult]],
    *,
    top_k: int = 3,
    rank_constant: int = 60,
) -> list[SearchResult]:
    """Merge ranked retrieval lists with reciprocal rank fusion.

    RRF rewards records that appear near the top of several result lists. The
    returned ``SearchResult.score`` is the fused score, not the original vector
    similarity score.
    """

    fused_scores: dict[str, float] = {}
    best_result_by_id: dict[str, SearchResult] = {}

    for results in ranked_result_sets:
        for rank, result in enumerate(results, start=1):
            record_id = result.record.id
            fused_scores[record_id] = fused_scores.get(record_id, 0.0) + 1.0 / (rank_constant + rank)
            if record_id not in best_result_by_id or result.score > best_result_by_id[record_id].score:
                best_result_by_id[record_id] = result

    fused_results = [
        SearchResult(record=best_result_by_id[record_id].record, score=score)
        for record_id, score in fused_scores.items()
    ]
    return sorted(fused_results, key=lambda result: result.score, reverse=True)[:top_k]


def retrieve_for_queries(
    queries: Sequence[str],
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    top_k: int = 3,
) -> list[list[SearchResult]]:
    """Run the baseline retriever once for each translated query."""

    return [retrieve(query, embedding_model, vector_store, top_k=top_k) for query in deduplicate_queries(queries)]


def retrieve_with_multi_query(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    translator: MultiQueryTranslator = simple_multi_queries,
    top_k: int = 3,
) -> list[SearchResult]:
    """Retrieve with query variants and merge results with RRF."""

    queries = translator(question)
    return reciprocal_rank_fusion(retrieve_for_queries(queries, embedding_model, vector_store, top_k=top_k), top_k=top_k)


def retrieve_with_decomposition(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    translator: MultiQueryTranslator = simple_decompose_query,
    top_k: int = 3,
) -> list[SearchResult]:
    """Retrieve for sub-questions and merge the evidence."""

    sub_questions = translator(question)
    return reciprocal_rank_fusion(
        retrieve_for_queries(sub_questions, embedding_model, vector_store, top_k=top_k),
        top_k=top_k,
    )


def retrieve_with_step_back(
    question: str,
    embedding_model: HashingEmbeddingModel,
    vector_store: InMemoryVectorStore,
    *,
    translator: QueryTranslator = simple_step_back_query,
    top_k: int = 3,
) -> list[SearchResult]:
    """Retrieve for the original and broader step-back queries, then fuse."""

    queries = [question, translator(question)]
    return reciprocal_rank_fusion(retrieve_for_queries(queries, embedding_model, vector_store, top_k=top_k), top_k=top_k)

