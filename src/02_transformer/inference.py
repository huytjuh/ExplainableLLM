"""Token generation utilities with traceable sampling decisions."""

from __future__ import annotations

import random
from dataclasses import dataclass

from attention import softmax


@dataclass(frozen=True)
class GenerationStep:
    step: int
    logits: list[float]
    probabilities: list[float]
    selected_token_id: int
    selected_token: str


def greedy_decode(logits: list[float]) -> int:
    return max(range(len(logits)), key=lambda index: logits[index])


def sample_next_token(
    logits: list[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> tuple[int, list[float]]:
    """Sample a token id and return the full probability vector used."""

    if temperature <= 0:
        token_id = greedy_decode(logits)
        probabilities = [0.0 for _ in logits]
        probabilities[token_id] = 1.0
        return token_id, probabilities

    scaled = [value / temperature for value in logits]
    if top_k is not None and top_k > 0 and top_k < len(scaled):
        keep = set(sorted(range(len(scaled)), key=lambda index: scaled[index], reverse=True)[:top_k])
        scaled = [value if index in keep else float("-inf") for index, value in enumerate(scaled)]

    probabilities = softmax(scaled)
    generator = rng or random.Random()
    draw = generator.random()
    cumulative = 0.0
    for token_id, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            return token_id, probabilities
    return len(probabilities) - 1, probabilities


def generate_tokens(
    next_logits,
    id_to_token: dict[int, str],
    *,
    max_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 7,
) -> list[GenerationStep]:
    """Generate tokens from a callback that returns logits for each step."""

    rng = random.Random(seed)
    steps: list[GenerationStep] = []
    generated_ids: list[int] = []

    for step in range(max_tokens):
        logits = next_logits(generated_ids)
        token_id, probabilities = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        generated_ids.append(token_id)
        steps.append(
            GenerationStep(
                step=step,
                logits=logits,
                probabilities=probabilities,
                selected_token_id=token_id,
                selected_token=id_to_token.get(token_id, "<unk>"),
            )
        )
    return steps
