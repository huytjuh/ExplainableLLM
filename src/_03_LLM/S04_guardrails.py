from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from S03_schemas import (
    LLMOutputValidationError,
    SentimentResult,
    TopicResult,
    parse_json_response,
    validate_sentiment_response,
    validate_topic_response,
)


class GuardrailViolation(ValueError):
    """Raised when input or output fails a deterministic guardrail."""


@dataclass(frozen=True)
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    errors: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise GuardrailViolation("; ".join(self.errors))


MIN_FEEDBACK_LENGTH = 3
MAX_FEEDBACK_LENGTH = 4_000
PROMPT_INJECTION_MARKERS = (
    "ignore previous instructions",
    "negeer vorige instructies",
    "negeer alle instructies",
    "system prompt",
    "developer message",
)


def check_feedback_text(feedback_text: str) -> GuardrailResult:
    """Check whether feedback text is safe and useful enough to send."""
    errors: list[str] = []
    normalized = feedback_text.strip()

    if len(normalized) < MIN_FEEDBACK_LENGTH:
        errors.append("Feedback text is too short.")
    if len(normalized) > MAX_FEEDBACK_LENGTH:
        errors.append("Feedback text is too long.")

    lowered = normalized.lower()
    for marker in PROMPT_INJECTION_MARKERS:
        if marker in lowered:
            errors.append(f"Feedback contains prompt-injection marker: {marker}")

    return GuardrailResult(passed=not errors, errors=errors)


def require_valid_feedback_text(feedback_text: str) -> str:
    """Validate feedback text and return the stripped text."""
    check_feedback_text(feedback_text).raise_if_failed()
    return feedback_text.strip()


def check_json_object_output(text: str) -> GuardrailResult:
    """Check whether an LLM output is parseable as a JSON object."""
    try:
        parse_json_response(text)
    except LLMOutputValidationError as error:
        return GuardrailResult(passed=False, errors=[str(error)])
    return GuardrailResult(passed=True)


def require_sentiment_output(text: str) -> SentimentResult:
    """Validate sentiment output or raise a guardrail violation."""
    try:
        return validate_sentiment_response(text)
    except LLMOutputValidationError as error:
        raise GuardrailViolation(str(error)) from error


def require_topic_output(
    text: str,
    topics_dir: str | Path = "config/topics",
) -> TopicResult:
    """Validate topic output against topic YAML files."""
    try:
        return validate_topic_response(text, allowed_topics=load_topic_labels(topics_dir))
    except LLMOutputValidationError as error:
        raise GuardrailViolation(str(error)) from error


def load_topic_labels(topics_dir: str | Path = "config/topics") -> set[str]:
    """Load allowed topic labels from split topic YAML files."""
    directory = Path(topics_dir)
    if not directory.is_absolute() and not directory.exists():
        directory = Path(__file__).resolve().parents[2] / directory

    labels: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        label = data.get("label")
        if isinstance(label, str) and label.strip():
            labels.add(label.strip())

    if not labels:
        raise GuardrailViolation(f"No topic labels found in {directory}")

    return labels


def should_retry_output(text: str) -> bool:
    """Return true when an LLM output looks repairable by a retry prompt."""
    return not check_json_object_output(text).passed
