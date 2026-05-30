"""LLM-as-a-judge prompt construction plus a local heuristic fallback."""

from __future__ import annotations

from dataclasses import dataclass

from metrics import token_f1


@dataclass(frozen=True)
class JudgeResult:
    score: float
    label: str
    rationale: str


def build_judge_prompt(question: str, answer: str, reference: str, context: str | None=None) -> str:
    return f"""You are evaluating an LLM answer.

Score from 1 to 5 using this rubric:
1 = incorrect or unsupported
3 = partially correct but incomplete
5 = correct, complete, and grounded

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{answer}

Retrieved context:
{context or ""}

Return JSON with score, label, and rationale."""


class HeuristicJudge:
    """A deterministic stand-in for an API-backed LLM judge."""

    def score(self, question: str, answer: str, reference: str, context: str | None=None) -> JudgeResult:
        del question
        f1 = token_f1(answer, reference)
        grounded_bonus = 0.1 if context and any(token in context.lower() for token in answer.lower().split()) else 0.0
        score = min(5.0, max(1.0, 1.0 + 4.0 * f1 + grounded_bonus))
        if score >= 4.0:
            label = "strong"
        elif score >= 2.5:
            label = "partial"
        else:
            label = "weak"
        return JudgeResult(score=round(score, 2), label=label, rationale=f"Token overlap F1={f1:.2f}.")
