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
    zeros,
)


@dataclass
class FastTextSkipGram:
    word_to_index: dict[str, int]
    index_to_word: list[str]
    ngram_to_index: dict[str, int]
    ngram_vectors: list[list[float]]
    output_vectors: list[list[float]]
    min_n: int
    max_n: int

    def vector(self, word: str) -> list[float]:
        ngrams = _char_ngrams(word, self.min_n, self.max_n)
        indices = [self.ngram_to_index[ngram] for ngram in ngrams if ngram in self.ngram_to_index]
        if not indices:
            return zeros(len(self.output_vectors[0]))
        vector = zeros(len(self.ngram_vectors[0]))
        for index in indices:
            add_scaled(vector, self.ngram_vectors[index], 1.0 / len(indices))
        return vector

    def similarity(self, left: str, right: str) -> float:
        return cosine_similarity(self.vector(left), self.vector(right))


def train_fasttext_skipgram(
    documents: list[str] | TokenizedCorpus,
    dimensions: int = 20,
    window_size: int = 2,
    epochs: int = 50,
    learning_rate: float = 0.05,
    negative_samples: int = 5,
    min_count: int = 1,
    min_n: int = 3,
    max_n: int = 6,
    seed: int = 13,
) -> FastTextSkipGram:
    """Train a tiny FastText-style subword skip-gram model from scratch."""
    corpus = tokenize_corpus(documents) if documents and isinstance(documents[0], str) else documents
    word_to_index, index_to_word = build_vocab(corpus, min_count=min_count)
    ngram_to_index = _build_ngram_vocab(index_to_word, min_n, max_n)
    rng = random.Random(seed)
    ngram_vectors = [random_vector(dimensions, rng) for _ in ngram_to_index]
    output_vectors = [random_vector(dimensions, rng) for _ in index_to_word]
    word_ngram_ids = [
        [ngram_to_index[ngram] for ngram in _char_ngrams(word, min_n, max_n)]
        for word in index_to_word
    ]
    pairs = context_pairs(corpus, word_to_index, window_size)
    negatives = negative_sampler(corpus, index_to_word)

    for _ in range(epochs):
        rng.shuffle(pairs)
        for center_id, context_id in pairs:
            _update_pair(
                ngram_vectors,
                output_vectors,
                word_ngram_ids[center_id],
                context_id,
                1,
                learning_rate,
            )
            for _ in range(negative_samples):
                negative_id = rng.choice(negatives)
                if negative_id != context_id:
                    _update_pair(
                        ngram_vectors,
                        output_vectors,
                        word_ngram_ids[center_id],
                        negative_id,
                        0,
                        learning_rate,
                    )

    return FastTextSkipGram(
        word_to_index,
        index_to_word,
        ngram_to_index,
        ngram_vectors,
        output_vectors,
        min_n,
        max_n,
    )


def _update_pair(
    ngram_vectors: list[list[float]],
    output_vectors: list[list[float]],
    center_ngram_ids: list[int],
    target_id: int,
    label: int,
    learning_rate: float,
) -> None:
    center = zeros(len(output_vectors[target_id]))
    for ngram_id in center_ngram_ids:
        add_scaled(center, ngram_vectors[ngram_id], 1.0 / len(center_ngram_ids))
    target = output_vectors[target_id]
    prediction = sigmoid(sum(c * t for c, t in zip(center, target)))
    gradient = learning_rate * (label - prediction)
    old_center = center[:]
    for ngram_id in center_ngram_ids:
        add_scaled(ngram_vectors[ngram_id], target, gradient / len(center_ngram_ids))
    add_scaled(target, old_center, gradient)


def _build_ngram_vocab(words: list[str], min_n: int, max_n: int) -> dict[str, int]:
    ngrams = sorted({ngram for word in words for ngram in _char_ngrams(word, min_n, max_n)})
    return {ngram: index for index, ngram in enumerate(ngrams)}


def _char_ngrams(word: str, min_n: int, max_n: int) -> list[str]:
    wrapped = f"<{word}>"
    ngrams: list[str] = []
    for size in range(min_n, max_n + 1):
        for start in range(0, len(wrapped) - size + 1):
            ngrams.append(wrapped[start : start + size])
    return ngrams or [wrapped]
