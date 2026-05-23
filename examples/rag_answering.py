import _path_setup  # noqa: F401

from S02_chunking import chunk_text
from S03_embeddings import HashingEmbeddingModel
from S04_indexing import InMemoryVectorStore
from S05_retrieval import retrieve
from S07_context import build_context
from recorder import TraceRecorder


if __name__ == "__main__":
    text = """
    ExplainableLLM teaches tokenization, transformer architecture, inference,
    RAG, vector search, tracing, evaluation, Gemini API usage, Vertex AI, and
    Azure DevOps artifacts.
    """
    question = "What does ExplainableLLM teach?"
    model = HashingEmbeddingModel()
    store = InMemoryVectorStore()

    for index, chunk in enumerate(chunk_text(text, chunk_size=20, overlap=5)):
        store.add(f"chunk-{index}", chunk, model.embed(chunk), {"source": "README-plan"})

    tracer = TraceRecorder("rag-demo")
    tracer.add_event("question", text=question)
    results = retrieve(question, model, store)
    context = build_context(results)
    answer = f"Based on the retrieved context, ExplainableLLM teaches: {context}"
    tracer.add_event("retrieval", result_count=len(results), context=context)
    tracer.add_event("answer", text=answer)
    trace_path = tracer.write()

    print(answer)
    print(f"Trace written to {trace_path}")
