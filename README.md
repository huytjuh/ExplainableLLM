# ExplainableLLM

ExplainableLLM is a developer- and researcher-focused project for learning large language models end to end. It explains the core LLM stack with code, math, and intuition: NLP model families, transformer internals, training objectives, optimization, inference, retrieval-augmented generation, vector search, evaluation, tracing, and LLMOps workflows.

The goal is implementation-level clarity. A reader should be able to follow how text becomes tokens, how tokens become vectors, how a transformer produces logits, how logits become final generated tokens, and how modern LLM applications are evaluated, traced, and delivered.

## Project Vision

This repository is a hands-on guide to modern LLM systems, from first principles to cloud-ready workflows.

It covers:

- Common NLP model families before LLMs.
- Tokenization, embeddings, attention, transformer blocks, training, optimization, and inference.
- Gemini 2.5 Flash Lite API usage.
- RAG pipelines with chunking, embeddings, vector databases, and vector search.
- LLM-as-a-judge evaluation.
- Tracing and observability for LLM applications.
- LLMOps workflows with Vertex AI, Azure DevOps, and pipeline artifacts.

## Learning Roadmap

### 1. Common NLP Models

- Rule-based NLP systems and linguistic pipelines.
- Bag-of-words, TF-IDF, and classical vector-space models.
- Naive Bayes, logistic regression, and support vector machines for text classification.
- Hidden Markov Models and Conditional Random Fields for sequence labeling.
- Word2Vec, GloVe, and FastText embeddings.
- Recurrent neural networks, LSTMs, and GRUs.
- Seq2seq models with attention.
- Encoder-only transformers such as BERT and RoBERTa.
- Decoder-only transformers such as GPT-style language models.
- Encoder-decoder transformers such as T5 and BART.
- Sentence embedding models for semantic search and clustering.

Deliverables:

- NLP model family map.
- Comparison table for classical, neural, and transformer-based models.
- Example tasks for classification, tagging, generation, retrieval, and ranking.
- Notes on when to use traditional NLP models versus modern LLMs.

### 2. Transformer Class: Foundations, Training, and Generation

This section combines the previous foundations, transformer architecture, training objectives, optimization, inference, and token generation sections into one end-to-end transformer learning path.

The detailed roadmap is preserved next to the implementation in `src/02_transformer/transformer.py`.

- Tokenization: text normalization, subwords, vocabularies, token IDs, BOS/EOS tokens.
- Embeddings and positional information: how token IDs become vectors with sequence order.
- Transformer architecture: self-attention, Q/K/V projections, causal masks, residual connections, feed-forward layers, layer normalization, and logits.
- Training objective: next-token prediction with cross-entropy loss and perplexity.
- Optimization: gradient descent, Adam/AdamW, learning-rate schedules, warmup, weight decay, and overfitting checks.
- Inference: prefill, decode, KV cache, greedy decoding, temperature, top-k/top-p sampling, stop sequences, streaming, and structured output.
- Final-token explanation: how logits become probabilities and how a decoding strategy selects each next token.

Deliverables:

- One transformer class walkthrough that connects tokenization, forward pass, loss, optimization, and generation.
- Small tokenizer implementation.
- Minimal decoder block implementation.
- Toy training loop and loss/perplexity examples.
- Token-by-token generation trace with logits, probabilities, and selected tokens.
- Sampling strategy demos.

### 3. LLM APIs with Gemini 2.5 Flash Lite

- Calling Gemini 2.5 Flash Lite through the API.
- Prompt construction.
- System instructions.
- Safety settings and response handling.
- Streaming.
- JSON and schema-constrained outputs.
- Cost, latency, and throughput considerations.

Deliverables:

- Gemini API client examples.
- Prompt templates.
- Structured output examples.
- Latency and cost measurement scripts.

### 4. Retrieval-Augmented Generation

- Why RAG is useful.
- Document ingestion.
- Chunking strategies.
- Embedding models.
- Metadata enrichment.
- Indexing.
- Retrieval.
- Reranking.
- Context assembly.
- Grounded answer generation.
- Citation and source attribution patterns.

Deliverables:

- End-to-end RAG pipeline.
- Chunking experiments.
- Retrieval quality evaluation.
- Grounded QA demo.

### 5. Vector Databases and Vector Search

- Dense vectors and similarity search.
- Cosine similarity, dot product, and Euclidean distance.
- Approximate nearest neighbor search.
- Index types and recall-latency tradeoffs.
- Metadata filtering.
- Hybrid search.
- Vector database options and selection criteria.
- Local development versus managed production stores.

Deliverables:

- Vector search demo.
- Similarity metric comparison.
- Vector database integration layer.
- Retrieval benchmark script.

### 6. Evaluation and LLM-as-a-Judge

- Exact-match and semantic evaluation.
- Faithfulness, relevance, groundedness, completeness, and helpfulness.
- LLM-as-a-judge prompt design.
- Pairwise comparisons.
- Rubrics.
- Bias and consistency risks.
- Human review loops.
- Regression testing for prompts and RAG pipelines.

Deliverables:

- Evaluation dataset format.
- Judge prompt templates.
- Automated scoring scripts.
- Evaluation report artifact.

### 7. Tracing and Observability

- Request tracing across prompt construction, retrieval, model calls, and post-processing.
- Token usage tracking.
- Latency breakdowns.
- Retrieval trace inspection.
- Prompt and response logging.
- Error handling and retry visibility.
- Experiment tracking.

Deliverables:

- Trace schema.
- Example traces for RAG and direct LLM calls.
- Debug dashboard concept.
- Observability checklist.

### 8. LLMOps

LLMOps is the production workflow layer around LLM applications. It uses traces, evaluations, reports, CI/CD, artifacts, and cloud integrations to make systems repeatable and inspectable.

- Vertex AI for model access, deployment, evaluation, and managed ML workflows.
- Gemini integration patterns on Google Cloud.
- Azure DevOps pipelines for CI/CD.
- Build artifacts for notebooks, evaluation reports, package outputs, and deployment bundles.
- Environment configuration and secret management.
- Automated tests for prompts, retrieval, and model outputs.

Deliverables:

- Vertex AI setup guide.
- Azure DevOps pipeline definition.
- Artifact publishing workflow.
- Deployment checklist.

### 9. Model Context Protocol

MCP connects LLM clients to tools, resources, prompts, and external systems through a standard protocol. This module comes last because it builds on the rest of the stack: model calls, RAG, vector search, evaluation, tracing, permissions, and LLMOps.

- MCP servers and client connections.
- Tools with typed inputs and outputs.
- Resources such as files, schemas, traces, and evaluation reports.
- Reusable prompts for workflows such as judging, debugging, and RAG query generation.
- Permission boundaries and operational safety.

Deliverables:

- MCP concept map.
- Example MCP server design for this repository.
- Tool/resource/prompt examples for LLM application workflows.
- Notes on how MCP fits with tracing, artifacts, and LLMOps.

## Repository Structure

```text
ExplainableLLM/
├── README.md
├── Makefile
├── pyproject.toml
├── poetry.lock
├── docs/
│   ├── 00-common-nlp-models.md
│   ├── 01-tokenization.md
│   ├── 02-transformers.md
│   ├── 03-training.md
│   ├── 04-inference.md
│   ├── 05-rag.md
│   ├── 06-vector-search.md
│   ├── 07-evaluation.md
│   ├── 08-tracing.md
│   └── 09-cloud-workflows.md
├── notebooks/
│   ├── tokenization_walkthrough.ipynb
│   ├── transformer_forward_pass.ipynb
│   ├── sampling_strategies.ipynb
│   └── rag_pipeline.ipynb
├── src/
│   ├── 01_NLP/
│   ├── 02_transformer/
│   ├── 03_LLM/
│   ├── 04_RAG/
│   │   ├── S01_documents.py
│   │   ├── S02_chunking.py
│   │   ├── S03_embeddings.py
│   │   ├── S04_indexing.py
│   │   ├── S05_retrieval.py
│   │   ├── S06_reranking.py
│   │   ├── S07_context.py
│   │   ├── S08_rag_chain.py
│   │   └── S09_citations.py
│   ├── 05_vector_search/
│   ├── 06_evaluation/
│   ├── 07_tracing/
│   ├── 08_LLMOps/
│   └── 09_MCP/
├── tests/
├── pipelines/
│   └── azure-pipelines.yml
├── artifacts/
│   ├── evaluations/
│   ├── traces/
│   └── reports/
└── examples/
    ├── basic_generation.py
    ├── rag_answering.py
    ├── vector_search.py
    ├── llm_as_judge.py
    └── gemini_flash_lite.py
```

## Quickstart

Install the project with Poetry:

```bash
poetry install
```

Run the tests:

```bash
poetry run pytest
```

Run the main quality check:

```bash
make check
```

Run local examples:

```bash
poetry run python examples/basic_generation.py
poetry run python examples/vector_search.py
poetry run python examples/rag_answering.py
poetry run python examples/llm_as_judge.py
```

Call Gemini 2.5 Flash Lite after setting credentials:

```bash
set GEMINI_API_KEY=your_api_key
set GEMINI_MODEL=gemini-2.5-flash-lite
poetry install --extras gemini
poetry run python examples/gemini_flash_lite.py
```

Provider settings are stored in YAML files under `config/`.
Use `config/gemini.yaml`, `config/chatgpt.yaml`, `config/claude.yaml`, or `config/ollama.yaml` as templates.
Do not store API keys in YAML files. Use environment variables instead:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_HOST`

The central client supports provider switching:

```python
from S01_client import LLMClient

client = LLMClient(provider="gemini")
client = LLMClient(provider="chatgpt")
client = LLMClient(provider="claude")
client = LLMClient(provider="llama")
client = LLMClient.from_config("config/gemini.local.yaml")
```

## Example End-to-End Flow

```text
User question
  -> Prompt builder
  -> Retriever
  -> Vector database
  -> Relevant chunks
  -> Context assembly
  -> Gemini 2.5 Flash Lite API
  -> Generated answer
  -> LLM-as-a-judge evaluation
  -> Trace and artifact storage
```

## Artifact Strategy

Artifacts make experiments reproducible and reviewable.

- `artifacts/evaluations/`: scoring outputs, judge results, benchmark summaries.
- `artifacts/traces/`: structured traces for LLM calls and RAG pipelines.
- `artifacts/reports/`: generated reports for model behavior, retrieval quality, and regressions.
- Pipeline artifacts: published outputs from Azure DevOps runs.

## Success Criteria

This project is successful when a reader can:

- Explain when to use classical NLP, transformer models, or LLM APIs.
- Explain how a prompt becomes tokens.
- Trace tensors through a transformer block.
- Understand how loss and optimization train a language model.
- Describe how logits become the next generated token.
- Build a basic RAG pipeline.
- Use vector search effectively.
- Call Gemini 2.5 Flash Lite through an API wrapper.
- Evaluate answers with an LLM-as-a-judge workflow.
- Inspect traces to debug model behavior.
- Run repeatable checks through Azure DevOps and publish artifacts.

## Intended Audience

- Developers building LLM applications.
- Researchers who want implementation-level intuition.
- ML engineers designing RAG and evaluation pipelines.
- Data scientists moving from notebooks to production workflows.
- Technical teams adopting Gemini, Vertex AI, vector databases, and Azure DevOps.

## License

License information will be added as the project evolves.
