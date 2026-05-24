from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class LLMOutputValidationError(ValueError):
    """Raised when an LLM response cannot be parsed or validated."""


class SentimentResult(BaseModel):
    """Validated output for Dutch customer sentiment classification."""

    sentiment: Literal["positief", "neutraal", "negatief"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return value.strip()


class FeedbackAnalysisResult(BaseModel):
    """Validated output for a fuller customer feedback analysis."""

    sentiment: Literal["positief", "neutraal", "negatief"]
    topic: str = Field(min_length=1)
    priority: Literal["laag", "middel", "hoog"]
    requires_follow_up: bool
    summary: str = Field(min_length=1)
    main_issue: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("topic", "summary", "main_issue", "recommended_action")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse an LLM response that should contain one JSON object."""
    cleaned = _strip_markdown_code_fence(text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise LLMOutputValidationError(
            f"LLM response is not valid JSON: {text}"
        ) from error

    if not isinstance(data, dict):
        raise LLMOutputValidationError("LLM response JSON must be an object.")

    return data


def validate_sentiment_response(text: str) -> SentimentResult:
    """Parse and validate a sentiment JSON response."""
    return _validate_model(parse_json_response(text), SentimentResult)


def validate_topic_response(
    text: str,
    allowed_topics: set[str] | None = None,
) -> TopicResult:
    """Parse and validate a topic JSON response."""
    result = _validate_model(parse_json_response(text), TopicResult)
    if allowed_topics is not None:
        result.validate_topic_label(allowed_topics)
    return result


def validate_feedback_analysis_response(text: str) -> FeedbackAnalysisResult:
    """Parse and validate a complete feedback analysis JSON response."""
    return _validate_model(parse_json_response(text), FeedbackAnalysisResult)


def _validate_model(data: dict[str, Any], model_type: type[BaseModel]) -> Any:
    try:
        return model_type.model_validate(data)
    except ValidationError as error:
        raise LLMOutputValidationError(str(error)) from error


def _strip_markdown_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text
