"""Tokenization examples for classic NLP preprocessing.

Tokenization is the step that turns raw text into a sequence of smaller units.
This example keeps punctuation as separate tokens so later preprocessing steps
can decide whether to keep or drop it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+(?:['-]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Tokenizer:
    """Small regex tokenizer for words, contractions, hyphenated terms, and punctuation."""

    lowercase: bool = True
    keep_punctuation: bool = True

    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens."""
        normalized = self.normalize(text)
        tokens = TOKEN_PATTERN.findall(normalized)

        if self.keep_punctuation:
            return tokens

        return [token for token in tokens if any(char.isalnum() for char in token)]

    def normalize(self, text: str) -> str:
        """Apply the tokenizer's normalization rules before splitting."""
        cleaned = " ".join(text.strip().split())
        return cleaned.lower() if self.lowercase else cleaned

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.tokenize("Customers can't log in to the mobile-app."))
