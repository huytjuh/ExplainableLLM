"""Simple answer-quality metrics."""

from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def exact_match(prediction: str, expected: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(expected))


def token_f1(prediction: str, expected: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(expected).split()
    if not predicted or not gold:
        return float(predicted == gold)

    overlap = set(predicted) & set(gold)
    if not overlap:
        return 0.0
    precision = len(overlap) / len(set(predicted))
    recall = len(overlap) / len(set(gold))
    return 2 * precision * recall / (precision + recall)

