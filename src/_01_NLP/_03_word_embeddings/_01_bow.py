from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BoWConfig:
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95

    stop_words: bool = False


class BagOfWords:
    """Simple Bag-of-Words embedding extractor."""

    def __init__(self, config: BoWConfig | None = None) -> None:
        """Initialize the BoW embedding extractor with the given configuration."""
        self.config = config or BoWConfig()
        self.vocabulary: list[str] = []

    def fit(self, corpus: list[Any]) -> 'BagOfWords':
        """Fit the BoW vocabulary to the provided documents."""
        tokens = [token for tokenized_text in corpus for token in tokenized_text.get_words(self.config.stop_words)]
        token_counts = Counter(tokens)

        max_df_threshold = int(self.config.max_df * len(corpus))
        self.vocabulary = [word for word, count in sorted(token_counts.items()) if count >= self.config.min_df and count <= max_df_threshold][:self.config.max_features]
        
        return self

    def transform(self, corpus: list[Any]) -> np.ndarray:
        """Transform documents into BoW embeddings using the fitted vocabulary."""
        if not self.vocabulary:
            raise ValueError("The BoW model must be fitted before transformation.")
        
        embeddings = np.zeros((len(corpus), len(self.vocabulary)), dtype=int)
        for i, tokenized_text in enumerate(corpus):
            tokens = set(tokenized_text.get_words(self.config.stop_words))
            embeddings[i] = [int(word in tokens) for word in self.vocabulary]
        
        return embeddings
    
    def fit_transform(self, corpus: list[Any]) -> np.ndarray:
        """Fit the vocabulary and return BoW embeddings for the same corpus."""
        return self.fit(corpus).transform(corpus)


@dataclass(frozen=True)
class _BagOfWordsCompat:
    vocabulary_: dict[str, int]

    def transform(self, documents: list[str]) -> list[list[int]]:
        rows: list[list[int]] = []
        for document in documents:
            counts = Counter(document.split())
            row = [0 for _ in self.vocabulary_]
            for word, index in self.vocabulary_.items():
                row[index] = counts[word]
            rows.append(row)
        return rows


def fit_bag_of_words(documents: list[str], min_count: int = 1) -> _BagOfWordsCompat:
    counts = Counter(token for document in documents for token in document.split())
    vocabulary = {
        word: index
        for index, word in enumerate(
            word for word, count in sorted(counts.items()) if count >= min_count
        )
    }
    return _BagOfWordsCompat(vocabulary)
    
