"""Deterministic local embeddings for examples and tests."""

from __future__ import annotations

import hashlib
import math
import re


class HashingEmbeddingModel:
    """A tiny hashing-vectorizer style embedding model.

    It is not semantic like a neural embedding model, but it is deterministic,
    dependency-free, and good enough to teach vector search mechanics.
    """

    def __init__(self, dimensions: int=64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

