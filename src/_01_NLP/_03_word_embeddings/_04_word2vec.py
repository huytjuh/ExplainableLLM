from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

@dataclass(frozen=True)
class NeuralNetwork:
    input_weights: np.ndarray
    output_weights: np.ndarray

@dataclass
class NeuralNetworkConfig:
    hidden_layer: list[int] = [20]
    hidden_activation: str = 'relu'
    output_activation: str = 'sigmoid'

    learning_rate: float = 0.05
    epochs: int = 50

@dataclass
class Word2VecConfig:
    method: Literal['skipgram', 'cbow'] = 'skipgram'
    negative_sampling: bool = True

    window_size: int = 1
    negative_samples: int = 5


class Word2Vec:
    """Simple Word2Vec extractor using skip-gram with negative sampling."""

    def __init__(self, config: Word2VecConfig | None = None, NNconfig: NeuralNetworkConfig | None = None) -> None:
        """Initialize the Word2Vec embedding extractor."""
        self.config = config or Word2VecConfig()
        self.vocabulary: list[str] = []
        self.word_idx: dict[str, int] = {}
        
        self.NNConfig = NNconfig or NeuralNetworkConfig()
        self.NN: NeuralNetwork | None = None

    def fit(self, corpus: list[Any]) -> Word2Vec:
        """Fit Word2Vec word vectors to the provided tokenized documents."""
        tokens = [token for tokenized_text in corpus for token in tokenized_text.get_words()]
        token_counts = Counter(tokens)

        max_df_threshold = int(self.config.max_df * len(corpus))
        self.vocabulary = [word for word, count in sorted(token_counts.items()) if count >= self.config.min_df and count <= max_df_threshold][:self.config.max_features]
        self.word_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        self._skipgram(corpus)

        return self
    
    def _skipgram(self, corpus: list[Any]) -> None:
        """Train the skip-gram model with negative sampling."""
        sent_tokens = [tokenized_text.get_sentences() for tokenized_text in corpus]
        sent_tokens = np.array([sent.get_words() for sent in sent_tokens])

        def context_pairs(sent_tokens: np.ndarray, window: int) -> list[tuple[int, int]]:

            


        tokenized_corpus = [tokenized_text.get_words() for tokenized_text in corpus]
        pairs = self._context_pairs(tokenized_corpus)
        token_counts = Counter(word for words in tokenized_corpus for word in words)
        negative_distribution = self._negative_distribution(token_counts)

        input_vectors = np.random.uniform(-0.5, 0.5, (len(self.vocabulary), self.config.vector_size))
        output_vectors = np.zeros((len(self.vocabulary), self.config.vector_size))

        for epoch in range(self.config.epochs):
            np.random.shuffle(pairs)
            for center_id, target_id in pairs:
                self._update_pair(center_id, target_id, label=1)
                if self.config.negative_sampling:
                    negative_ids = np.random.choice(len(self.vocabulary), size=self.config.negative_samples, p=negative_distribution)
                    for negative_id in negative_ids:
                        self._update_pair(center_id, negative_id, label=0)

        self.NN = NeuralNetwork(input_weights=input_vectors, output_weights=output_vectors)

    def transform(self, corpus: list[Any]) -> np.ndarray:
        """Transform documents into averaged Word2Vec document embeddings."""
        if self.NN is None:
            raise ValueError("The Word2Vec model must be fitted before transformation.")

        embeddings: list[np.ndarray] = []
        for tokenized_text in corpus:
            vectors = [
                self.input_vectors[self.word_to_index[token]]
                for token in _get_words(tokenized_text)
                if token in self.word_to_index
            ]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.config.vector_size))

        return np.vstack(embeddings)

    def fit_transform(self, corpus: list[Any]) -> np.ndarray:
        """Fit the model and return averaged document embeddings."""
        return self.fit(corpus).transform(corpus)
