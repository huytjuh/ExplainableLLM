"""Common NLP model family reference data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    name: str
    examples: tuple[str, ...]
    typical_tasks: tuple[str, ...]
    notes: str


def model_family_map() -> list[ModelFamily]:
    return [
        ModelFamily(
            name="Rules and patterns",
            examples=("regular expressions", "dictionaries", "linguistic rules"),
            typical_tasks=("validation", "deterministic extraction"),
            notes="Fast and predictable, but brittle outside expected language.",
        ),
        ModelFamily(
            name="Sparse vectors",
            examples=("bag-of-words", "TF-IDF"),
            typical_tasks=("search", "clustering", "classification baselines"),
            notes="Simple and strong for baselines, but weak on semantics and word order.",
        ),
        ModelFamily(
            name="Classical ML",
            examples=("Naive Bayes", "logistic regression", "SVM"),
            typical_tasks=("text classification", "ranking"),
            notes="Efficient when labels are stable and feature engineering is acceptable.",
        ),
        ModelFamily(
            name="Sequence models",
            examples=("HMM", "CRF", "LSTM", "GRU"),
            typical_tasks=("tagging", "entity extraction", "sequence prediction"),
            notes="Useful for sequence structure, though transformers handle long context better.",
        ),
        ModelFamily(
            name="Transformers",
            examples=("BERT", "GPT", "T5", "BART"),
            typical_tasks=("understanding", "generation", "retrieval", "ranking"),
            notes="The dominant modern architecture for LLMs and many NLP tasks.",
        ),
    ]

