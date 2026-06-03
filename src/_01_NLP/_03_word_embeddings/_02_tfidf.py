from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TfidfConfig:
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95

    stop_words: bool = False

    smooth_idf: bool = True


class Tfidf:
    """Simple TF-IDF embedding extractor."""

    def __init__(self, config: TfidfConfig | None = None) -> None:
        """Initialize the TF-IDF embedding extractor with the given configuration."""
        self.config = config or TfidfConfig()

        self.vocabulary: list[str] = []
        self.idf: np.ndarray | None = None
        
    def fit(self, corpus: list[Any]) -> 'Tfidf':
        """Fit the TF-IDF model to the provided documents."""
        tokens = [token for tokenized_text in corpus for token in tokenized_text.get_words(self.config.stop_words)]
        token_counts = Counter(tokens)

        max_df_threshold = int(self.config.max_df * len(corpus))
        self.vocabulary = [word for word, count in sorted(token_counts.items()) if count >= self.config.min_df and count <= max_df_threshold][:self.config.max_features]

        # INVERSE DOCUMENT FREQUENCY
        doc_counts = Counter()
        for tokenized_text in corpus:
            doc_counts.update(tokenized_text.get_words(self.config.stop_words))
        document_frequency = np.array([doc_counts[word] for word in self.vocabulary], dtype=int)

        self.idf = self._inverse_document_frequency(document_frequency, len(corpus))

        return self 
    
    def _inverse_document_frequency(self, document_frequency: np.ndarray, n_documents: int) -> np.ndarray:
        """Calculate the inverse document frequency for each token in the vocabulary."""
        if self.config.smooth_idf:
            document_frequency += 1
            n_documents += 1
            
        return np.log((n_documents) / document_frequency) + 1.0
    
    def _term_frequency(self, tokens: list[str]) -> np.ndarray:
        """Calculate the term frequency for each token in the vocabulary."""
        token_counts = Counter(tokens)
        return np.array([token_counts[word] / len(tokens) for word in self.vocabulary], dtype=float)

    def transform(self, corpus: list[Any]) -> np.ndarray:
        """Transform documents into TF-IDF embeddings using the fitted model."""
        if not self.vocabulary or self.idf is None:
            raise ValueError("The TF-IDF model must be fitted before transformation.")
        
        # TERM FREQUENCY * INVERSE DOCUMENT FREQUENCY
        embeddings = np.zeros((len(corpus), len(self.vocabulary)), dtype=float)
        for i, tokenized_text in enumerate(corpus):
            tokens = tokenized_text.get_words(self.config.stop_words)
            tf = self._term_frequency(tokens)
            embeddings[i] = tf * self.idf
        
        return embeddings

    def fit_transform(self, corpus: list[Any]) -> np.ndarray:
        """Fit the model and return TF-IDF embeddings for the same corpus."""
        return self.fit(corpus).transform(corpus)
