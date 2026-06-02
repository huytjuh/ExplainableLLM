from __future__ import annotations

from dataclasses import dataclass
import random

from _common import (
    TokenizedCorpus,
    add_scaled,
    build_vocab,
    context_pairs,
    cosine_similarity,
    negative_sampler,
    random_vector,
    sigmoid,
    tokenize_corpus,
)


@dataclass
class Word2VecSkipGram:
    word_to_index: dict[str, int]
    index_to_word: list[str]
    input_vectors: list[list[float]]
    output_vectors: list[list[float]]

    def vector(self, word: str) -> list[float]:
        return self.input_vectors[self.word_to_index[word]][:]

    def similarity(self, left: str, right: str) -> float:
        return cosine_similarity(self.vector(left), self.vector(right))


def train_word2vec_skipgram(
    documents: list[str] | TokenizedCorpus,
    dimensions: int = 20,
    window_size: int = 2,
    epochs: int = 50,
    learning_rate: float = 0.05,
    negative_samples: int = 5,
    min_count: int = 1,
    seed: int = 13,
) -> Word2VecSkipGram:
    """Train a tiny skip-gram with negative sampling model from scratch."""
    corpus = tokenize_corpus(documents) if documents and isinstance(documents[0], str) else documents
    word_to_index, index_to_word = build_vocab(corpus, min_count=min_count)
    rng = random.Random(seed)
    input_vectors = [random_vector(dimensions, rng) for _ in index_to_word]
    output_vectors = [random_vector(dimensions, rng) for _ in index_to_word]
    pairs = context_pairs(corpus, word_to_index, window_size)
    negatives = negative_sampler(corpus, index_to_word)

    for _ in range(epochs):
        rng.shuffle(pairs)
        for center_id, context_id in pairs:
            _update_pair(input_vectors, output_vectors, center_id, context_id, 1, learning_rate)
            for _ in range(negative_samples):
                negative_id = rng.choice(negatives)
                if negative_id != context_id:
                    _update_pair(input_vectors, output_vectors, center_id, negative_id, 0, learning_rate)

    return Word2VecSkipGram(word_to_index, index_to_word, input_vectors, output_vectors)


def _update_pair(
    input_vectors: list[list[float]],
    output_vectors: list[list[float]],
    center_id: int,
    target_id: int,
    label: int,
    learning_rate: float,
) -> None:
    center = input_vectors[center_id]
    target = output_vectors[target_id]
    prediction = sigmoid(sum(c * t for c, t in zip(center, target)))
    gradient = learning_rate * (label - prediction)
    old_center = center[:]
    add_scaled(center, target, gradient)
    add_scaled(target, old_center, gradient)
