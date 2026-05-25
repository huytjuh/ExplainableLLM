from S02_chunking import chunk_text
from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore, RAGIndexer, WordChunker
from S05_retrieval import retrieve
from S07_context import build_context


def test_chunk_text_with_overlap():
    chunks = chunk_text("one two three four five six", chunk_size=3, overlap=1)
    assert chunks == ["one two three", "three four five", "five six"]


def test_vector_search_retrieves_relevant_chunk():
    model = HashingEmbeddingModel(dimensions=32)
    store = InMemoryVectorStore()
    store.add("rag", "RAG retrieves context for LLM answers.", model.embed("RAG retrieves context for LLM answers."))
    store.add("training", "Cross entropy trains next-token prediction.", model.embed("Cross entropy trains next-token prediction."))

    results = retrieve("How does retrieval help answer generation?", model, store, top_k=1)

    assert len(results) == 1
    assert results[0].record.id in {"rag", "training"}
    assert "score=" in build_context(results)


def test_rag_indexer_loads_chunks_embeds_and_searches_directory(tmp_path):
    (tmp_path / "billing.txt").write_text(
        "The customer cancelled a subscription but still received a billing charge and invoice.",
        encoding="utf-8",
    )
    (tmp_path / "delivery.txt").write_text(
        "The order arrived late after the tracking page stopped updating.",
        encoding="utf-8",
    )

    indexer = RAGIndexer(
        chunker=WordChunker(chunk_size=12, overlap=3),
        embedding_model=HashingEmbeddingModel(dimensions=256),
    )

    indexed_chunks = indexer.index_directory(tmp_path)
    results = indexer.search("subscription billing invoice", top_k=1)

    assert indexed_chunks >= 2
    assert len(results) == 1
    assert results[0].record.metadata["document_id"] == "billing"
