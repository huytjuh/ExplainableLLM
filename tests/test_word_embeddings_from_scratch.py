from __future__ import annotations

from pathlib import Path
import sys


EMBEDDING_DIR = Path(__file__).resolve().parents[1] / "src" / "_01_NLP" / "_03_word_embeddings"
sys.path.insert(0, str(EMBEDDING_DIR))

from _01_bow import fit_bag_of_words
from _02_tfidf import fit_tfidf
from _03_word2vec import train_word2vec_skipgram
from _04_glove import train_glove
from _05_fasttext import train_fasttext_skipgram


CORPUS = [
    "delivery delay refund",
    "delivery delay support",
    "refund payment support",
    "mobile app payment",
]


def test_bag_of_words_counts_terms():
    model = fit_bag_of_words(CORPUS)

    matrix = model.transform(["delivery delivery refund"])

    assert matrix[0][model.vocabulary_["delivery"]] == 2
    assert matrix[0][model.vocabulary_["refund"]] == 1


def test_tfidf_downweights_common_terms():
    model = fit_tfidf(CORPUS, normalize=False)

    row = model.transform(["delivery mobile"])[0]

    assert row[model.vocabulary_["mobile"]] > row[model.vocabulary_["delivery"]]


def test_word2vec_trains_dense_word_vectors():
    model = train_word2vec_skipgram(CORPUS, dimensions=8, epochs=3, negative_samples=2)

    assert len(model.vector("delivery")) == 8
    assert -1.0 <= model.similarity("delivery", "refund") <= 1.0


def test_glove_trains_dense_word_vectors():
    model = train_glove(CORPUS, dimensions=8, epochs=3)

    assert len(model.vector("support")) == 8
    assert -1.0 <= model.similarity("support", "payment") <= 1.0


def test_fasttext_uses_subwords_for_unseen_forms():
    model = train_fasttext_skipgram(CORPUS, dimensions=8, epochs=3, negative_samples=2)

    vector = model.vector("payments")

    assert len(vector) == 8
    assert any(value != 0.0 for value in vector)
