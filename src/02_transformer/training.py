"""Training-objective helpers for explanation notebooks and tests."""

from __future__ import annotations

import math


def cross_entropy_loss(probabilities: list[float], target_index: int, epsilon: float=1e-12) -> float:
    """Cross-entropy for one next-token target."""

    if target_index < 0 or target_index >= len(probabilities):
        raise IndexError("target_index is outside the probability vector")
    probability = max(probabilities[target_index], epsilon)
    return -math.log(probability)


def sequence_cross_entropy(probability_rows: list[list[float]], target_indices: list[int]) -> float:
    if len(probability_rows) != len(target_indices):
        raise ValueError("probability_rows and target_indices must have the same length")
    losses = [cross_entropy_loss(row, target) for row, target in zip(probability_rows, target_indices)]
    return sum(losses) / len(losses)


def perplexity(loss: float) -> float:
    return math.exp(loss)


def adamw_update(
    parameter: float,
    gradient: float,
    *,
    learning_rate: float,
    weight_decay: float=0.01,
) -> float:
    """A single educational AdamW-like step without moment history."""

    return parameter - learning_rate * (gradient + weight_decay * parameter)

