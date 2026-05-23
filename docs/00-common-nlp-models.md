# Common NLP Models

This section gives the map before the deep dive. LLMs are part of a long line of NLP systems, and many production applications still combine older and newer techniques.

## Model Families

| Family | Examples | Good For | Main Limitation |
| --- | --- | --- | --- |
| Rules and patterns | regex, dictionaries, linguistic rules | deterministic extraction, validation | brittle outside expected inputs |
| Sparse vectors | bag-of-words, TF-IDF | search, clustering, simple classification | weak semantics and word order |
| Classical ML | Naive Bayes, logistic regression, SVM | fast text classification | needs feature engineering |
| Sequence models | HMM, CRF | tagging, entity extraction | limited long-range context |
| Word embeddings | Word2Vec, GloVe, FastText | similarity, features for downstream models | static meaning per word |
| RNNs | RNN, LSTM, GRU | sequence modeling | harder to parallelize |
| Seq2seq | encoder-decoder with attention | translation, summarization | less scalable than transformers |
| Transformers | BERT, GPT, T5, BART | understanding, generation, retrieval, ranking | compute and data hungry |
| Sentence embeddings | SBERT-style models | semantic search, deduplication | depends on embedding domain fit |

## Practical Rule

Use the simplest model that satisfies the task. A TF-IDF baseline is often worth building before a RAG pipeline. A small classifier may be better than an LLM when the label space is stable, latency is strict, and explanations need to be deterministic.

