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

## Word Representation Models

The from-scratch examples live in `src/_01_NLP/_03_word_embeddings/`.

| Model | Representation | Main Difference | Pros | Cons |
| --- | --- | --- | --- | --- |
| Bag of Words | Sparse document vector of raw token counts | Counts words independently; no weighting by importance | Simple, deterministic, interpretable, strong baseline for small classifiers | Ignores order and semantics; frequent filler words can dominate |
| TF-IDF | Sparse document vector of term frequency times inverse document frequency | Keeps BoW shape but downweights terms that appear in many documents | Good baseline for search, clustering, and classification; still interpretable | No true semantic similarity; vocabulary is fixed; word order is lost |
| Word2Vec | Dense word vector learned from local context windows | Predictive model: learns words that occur in similar contexts | Captures useful semantic similarity; efficient after training | Static meaning per word; cannot represent unseen words; needs enough corpus signal |
| GloVe | Dense word vector learned from global co-occurrence counts | Count-based objective over the corpus-level co-occurrence matrix | Uses global statistics well; stable and interpretable training objective | Co-occurrence matrix can be large; static meaning; weak on rare/unseen words |
| FastText | Dense word vector built from character n-gram vectors | Extends Word2Vec with subword units | Handles misspellings, morphology, and some unseen words better | More memory/training work than Word2Vec; subword overlap can create noisy similarities |

BoW and TF-IDF create document features. Word2Vec, GloVe, and FastText create word embeddings that can be averaged or pooled into document features, or used directly for nearest-neighbor word similarity.
