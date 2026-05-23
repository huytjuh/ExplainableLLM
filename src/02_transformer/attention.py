"""Readable attention primitives using only the Python standard library."""

from __future__ import annotations

import math

Vector = list[float]
Matrix = list[Vector]


def dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right))


def softmax(values: Vector) -> Vector:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def weighted_sum(weights: Vector, values: Matrix) -> Vector:
    if not values:
        return []
    width = len(values[0])
    return [sum(weight * row[i] for weight, row in zip(weights, values)) for i in range(width)]


def scaled_dot_product_attention(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    *,
    causal: bool=True,
) -> tuple[Matrix, Matrix]:
    """Return attention outputs and the attention weight matrix."""

    if not queries or not keys or not values:
        return [], []

    scale = math.sqrt(len(keys[0]))
    outputs: Matrix=[]
    all_weights: Matrix=[]

    for query_index, query in enumerate(queries):
        scores = []
        for key_index, key in enumerate(keys):
            if causal and key_index > query_index:
                scores.append(float("-inf"))
            else:
                scores.append(dot(query, key) / scale)
        weights = softmax(scores)
        outputs.append(weighted_sum(weights, values))
        all_weights.append(weights)

    return outputs, all_weights

