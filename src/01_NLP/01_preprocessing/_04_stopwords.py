"""Stopword removal example.

Stopwords are frequent function words that are often removed in bag-of-words
and retrieval pipelines. They should usually be kept for tasks where word order
or exact wording matters, such as generation or sentiment with negation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}

DUTCH_STOPWORDS = {
    "aan",
    "als",
    "bij",
    "dat",
    "de",
    "een",
    "en",
    "het",
    "ik",
    "in",
    "is",
    "met",
    "op",
    "te",
    "van",
    "voor",
}


@dataclass(frozen=True)
class StopwordRemover:
    """Remove common words from token sequences."""

    language: str = "english"
    extra_stopwords: set[str] = field(default_factory=set)
    keep_negations: bool = True

    def __post_init__(self) -> None:
        if self.language not in {"english", "dutch", "both"}:
            raise ValueError("language must be 'english', 'dutch', or 'both'.")

    @property
    def stopwords(self) -> set[str]:
        words = set()
        if self.language in {"english", "both"}:
            words.update(ENGLISH_STOPWORDS)
        if self.language in {"dutch", "both"}:
            words.update(DUTCH_STOPWORDS)
        words.update(word.lower() for word in self.extra_stopwords)

        if self.keep_negations:
            words.difference_update({"no", "not", "niet", "geen"})

        return words

    def remove(self, tokens: list[str]) -> list[str]:
        """Drop stopwords while preserving punctuation and content words."""
        stopwords = self.stopwords
        return [token for token in tokens if token.lower() not in stopwords]

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.remove(tokens)


if __name__ == "__main__":
    remover = StopwordRemover(language="both")
    print(remover.remove(["the", "delivery", "is", "not", "on", "time"]))
