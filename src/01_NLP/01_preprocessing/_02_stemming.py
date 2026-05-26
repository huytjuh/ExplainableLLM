"""Rule-based stemming example.

Stemming reduces related word forms to a rough shared root. Unlike a real
Porter or Snowball stemmer, this class deliberately uses visible suffix rules
so the mechanics are easy to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_SUFFIXES = (
    "ization",
    "ational",
    "fulness",
    "ousness",
    "iveness",
    "tional",
    "heden",
    "eren",
    "ing",
    "ers",
    "ies",
    "ied",
    "ed",
    "en",
    "er",
    "es",
    "s",
)


@dataclass(frozen=True)
class RuleBasedStemmer:
    """Apply simple suffix stripping to individual tokens or token sequences."""

    suffixes: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SUFFIXES)
    min_stem_length: int = 3

    def stem(self, token: str) -> str:
        """Return a rough stem for one token."""
        if not token.isalpha():
            return token

        lowered = token.lower()
        for suffix in self.suffixes:
            if self._can_strip(lowered, suffix):
                return self._normalize_suffix(lowered.removesuffix(suffix), suffix)

        return lowered

    def transform(self, tokens: list[str]) -> list[str]:
        """Stem each token in a tokenized sentence."""
        return [self.stem(token) for token in tokens]

    def _can_strip(self, token: str, suffix: str) -> bool:
        return token.endswith(suffix) and len(token) - len(suffix) >= self.min_stem_length

    @staticmethod
    def _normalize_suffix(stem: str, suffix: str) -> str:
        if suffix in {"ies", "ied"}:
            return f"{stem}y"
        return stem

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.transform(tokens)


if __name__ == "__main__":
    stemmer = RuleBasedStemmer()
    print(stemmer.transform(["customers", "cancelled", "deliveries", "betaling"]))
