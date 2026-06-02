from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TfidfModel:
    vocabulary_: dict[str, int]
    feature_names_: list[str]
    idf_: list[float]
    normalize: bool = True

    def transform(self, documents: list[str] | TokenizedCorpus) -> list[list[float]]:
        corpus = tokenize_corpus(documents) if documents and isinstance(documents[0], str) else documents
        rows: list[list[float]] = []
        for tokens in corpus:
            counts = Counter(token for token in tokens if token in self.vocabulary_)
            total = sum(counts.values()) or 1
            row = [0.0 for _ in self.feature_names_]
            for token, count in counts.items():
                index = self.vocabulary_[token]
                tf = count / total
                row[index] = tf * self.idf_[index]
            rows.append(l2_normalize(row) if self.normalize else row)
        return rows


def fit_tfidf(
    documents: list[str] | TokenizedCorpus,
    min_count: int = 1,
    smooth_idf: bool = True,
    normalize: bool = True,
) -> TfidfModel:
    """Fit term-frequency inverse-document-frequency vectors from scratch."""
    corpus = tokenize_corpus(documents) if documents and isinstance(documents[0], str) else documents
    vocabulary, feature_names = build_vocab(corpus, min_count=min_count)
    document_frequency = [0 for _ in feature_names]
    for tokens in corpus:
        seen = {token for token in tokens if token in vocabulary}
        for token in seen:
            document_frequency[vocabulary[token]] += 1

    n_documents = len(corpus)
    idf: list[float] = []
    for df in document_frequency:
        if smooth_idf:
            idf.append(math.log((1 + n_documents) / (1 + df)) + 1.0)
        else:
            idf.append(math.log(n_documents / df) if df else 0.0)
    return TfidfModel(vocabulary, feature_names, idf, normalize=normalize)
