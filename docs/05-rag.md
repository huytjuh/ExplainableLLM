# Retrieval-Augmented Generation

RAG adds external knowledge to an LLM prompt.

Pipeline:

1. Load documents.
2. Split documents into chunks.
3. Embed each chunk.
4. Store vectors and metadata.
5. Embed the user question.
6. Retrieve relevant chunks with vector search.
7. Assemble context.
8. Ask the LLM to answer using the context.
9. Trace and evaluate the result.

