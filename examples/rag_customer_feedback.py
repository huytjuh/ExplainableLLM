import _path_setup  # noqa: F401

from pathlib import Path

from S03_embeddings import HashingEmbeddingModel
from S04_indexing import RAGIndexer, WordChunker
from S07_context import build_context


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_DIR = ROOT / "data" / "rag_customer_feedback"


if __name__ == "__main__":
    question = "subscription billing charge invoice"
    indexer = RAGIndexer(
        chunker=WordChunker(chunk_size=80, overlap=15),
        embedding_model=HashingEmbeddingModel(dimensions=4096),
    )
    indexer.index_directory(DOCUMENT_DIR)

    results = indexer.search(question, top_k=4)
    context = build_context(results)

    print(f"Question: {question}\n")
    print("Retrieved context:\n")
    print(context)
    print("\nSources:")
    for result in results:
        source = result.record.metadata["source"]
        print(f"- {source} score={result.score:.3f}")
