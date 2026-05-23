import _path_setup  # noqa: F401

from chunking import chunk_text
from embeddings import HashingEmbeddingModel
from recorder import TraceRecorder
from retrieve import build_context, retrieve
from vector_store import InMemoryVectorStore


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
