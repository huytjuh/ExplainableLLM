from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BoWConfig:
    min_frequency: int = 1
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
        dict_counts = Counter(tokens)

        self.vocabulary = [word for word, count in sorted(dict_counts.items()) if count >= self.config.min_frequency]
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
    
