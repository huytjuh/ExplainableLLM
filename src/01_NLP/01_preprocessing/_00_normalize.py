"""Text normalization examples for NLP preprocessing.

Normalization makes noisy raw text more consistent before tokenization,
vectorization, search, or model input. This module keeps the rules explicit so
they are easy to adjust for a specific dataset.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_PATTERN = re.compile(r"\s+")

@dataclass(frozen=True)
class TextNormalizer:
    """Normalize raw text into a predictable preprocessing format."""

    lowercase: bool = True
    strip_accents: bool = False
    collapse_whitespace: bool = True
    remove_control_chars: bool = True
    unicode_form: str = "NFKC"

    def normalize(self, text: str) -> str:
        """Return normalized text using the configured rules."""
        normalized = unicodedata.normalize(self.unicode_form, text)

        if self.remove_control_chars:
            normalized = CONTROL_PATTERN.sub(" ", normalized)

        if self.strip_accents:
            normalized = self._strip_accents(normalized)

        if self.lowercase:
            normalized = normalized.lower()

        normalized = normalized.strip()
        if self.collapse_whitespace:
            normalized = WHITESPACE_PATTERN.sub(" ", normalized)

        return normalized

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    @staticmethod
    def _strip_accents(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(text: str, **kwargs: object) -> str:
    """Convenience wrapper for one-off normalization."""
    return TextNormalizer(**kwargs).normalize(text)
