from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class NormalizedText:
    text: str


@dataclass
class NormalizerConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    collapse_whitespace: bool = True


class Normalizer:
    """Normalize text by lowercasing, removing punctuation, and collapsing whitespace."""

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        """Initialize the Normalizer with the given configuration."""
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> NormalizedText:
        """Normalize the input text according to the configuration."""
        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            text = re.sub(r"[!?]+", ".", text)
            text = re.sub(r"\.{2,}", ".", text)
            text = re.sub(r";+", ".", text)
            text = re.sub(r"[\\/]+", " ", text)
            text = re.sub(r"[\"()\[\]{}<>]", "", text)

        if self.config.collapse_whitespace:
            text = re.sub(r"\s+", " ", text)

        return NormalizedText(text=text.strip())

    def __call__(self, text: str) -> NormalizedText:
        """Allow the Normalizer to be called directly on text."""
        return self.normalize(text)