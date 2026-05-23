import _path_setup  # noqa: F401

from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore


if __name__ == "__main__":
    model = HashingEmbeddingModel(dimensions=32)
    store = InMemoryVectorStore()
    documents = {
        "tokenization": "Tokenization maps text into token ids.",
        "rag": "RAG retrieves context before asking an LLM to answer.",
        "judge": "LLM-as-a-judge evaluates answer quality with a rubric.",
    }

    for doc_id, text in documents.items():
        store.add(doc_id, text, model.embed(text), {"source": doc_id})

    for result in store.search(model.embed("How does retrieval help LLM answers?"), top_k=2):
        print(f"{result.score:.3f} {result.record.id}: {result.record.text}")
