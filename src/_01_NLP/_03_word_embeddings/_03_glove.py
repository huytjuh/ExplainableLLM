from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
import random

from _common import TokenizedCorpus, build_vocab, cosine_similarity, random_vector, tokenize_corpus


@dataclass
class GloveModel:
    word_to_index: dict[str, int]
    index_to_word: list[str]
    word_vectors: list[list[float]]
    context_vectors: list[list[float]]
    word_biases: list[float]
    context_biases: list[float]

    def vector(self, word: str) -> list[float]:
        index = self.word_to_index[word]
        return [
            left + right
            for left, right in zip(self.word_vectors[index], self.context_vectors[index])
        ]

    def similarity(self, left: str, right: str) -> float:
        return cosine_similarity(self.vector(left), self.vector(right))


def train_glove(
    documents: list[str] | TokenizedCorpus,
    dimensions: int = 20,
    window_size: int = 2,
    epochs: int = 50,
    learning_rate: float = 0.03,
    min_count: int = 1,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 13,
) -> GloveModel:
    """Train a small GloVe-style weighted least-squares model from scratch."""
    corpus = tokenize_corpus(documents) if documents and isinstance(documents[0], str) else documents
    word_to_index, index_to_word = build_vocab(corpus, min_count=min_count)
    cooccurrences = _cooccurrence_counts(corpus, word_to_index, window_size)
    rng = random.Random(seed)
    word_vectors = [random_vector(dimensions, rng) for _ in index_to_word]
    context_vectors = [random_vector(dimensions, rng) for _ in index_to_word]
    word_biases = [0.0 for _ in index_to_word]
    context_biases = [0.0 for _ in index_to_word]
    items = list(cooccurrences.items())

    for _ in range(epochs):
        rng.shuffle(items)
        for (word_id, context_id), count in items:
            weight = (count / x_max) ** alpha if count < x_max else 1.0
            prediction = (
                sum(a * b for a, b in zip(word_vectors[word_id], context_vectors[context_id]))
                + word_biases[word_id]
                + context_biases[context_id]
            )
            error = weight * (prediction - math.log(count))
            old_word = word_vectors[word_id][:]
            for dim in range(dimensions):
                word_vectors[word_id][dim] -= learning_rate * error * context_vectors[context_id][dim]
                context_vectors[context_id][dim] -= learning_rate * error * old_word[dim]
            word_biases[word_id] -= learning_rate * error
            context_biases[context_id] -= learning_rate * error

    return GloveModel(
        word_to_index,
        index_to_word,
        word_vectors,
        context_vectors,
        word_biases,
        context_biases,
    )


def _cooccurrence_counts(
    corpus: TokenizedCorpus,
    word_to_index: dict[str, int],
    window_size: int,
) -> dict[tuple[int, int], float]:
    counts: dict[tuple[int, int], float] = defaultdict(float)
    for document in corpus:
        ids = [word_to_index[token] for token in document if token in word_to_index]
        for center_pos, center_id in enumerate(ids):
            start = max(0, center_pos - window_size)
            stop = min(len(ids), center_pos + window_size + 1)
            for context_pos in range(start, stop):
                if context_pos == center_pos:
                    continue
                distance = abs(center_pos - context_pos)
                counts[(center_id, ids[context_pos])] += 1.0 / distance
    return dict(counts)
