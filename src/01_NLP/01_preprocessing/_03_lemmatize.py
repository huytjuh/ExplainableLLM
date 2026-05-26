"""Rule-based lemmatization example.

Lemmatization maps inflected forms to dictionary-like base forms. Production
lemmatizers usually need part-of-speech tags and language-specific resources;
this example uses a small lexicon plus transparent fallback rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_LEXICON = {
    "am": "be",
    "are": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "better": "good",
    "best": "good",
    "bought": "buy",
    "children": "child",
    "customers": "customer",
    "delivered": "deliver",
    "deliveries": "delivery",
    "mice": "mouse",
    "orders": "order",
    "slechter": "slecht",
    "terugbetalingen": "terugbetaling",
}


@dataclass(frozen=True)
class Lemmatizer:
    """Map tokens to lemmas with a lookup table and simple fallback rules."""

    lexicon: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LEXICON))

    def lemmatize(self, token: str) -> str:
        """Return the lemma for one token."""
        if not token.isalpha():
            return token

        lowered = token.lower()
        if lowered in self.lexicon:
            return self.lexicon[lowered]

        return self._fallback_lemma(lowered)

    def transform(self, tokens: list[str]) -> list[str]:
        """Lemmatize each token in a tokenized sentence."""
        return [self.lemmatize(token) for token in tokens]

    @staticmethod
    def _fallback_lemma(token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return f"{token[:-3]}y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def __call__(self, tokens: list[str]) -> list[str]:
        return self.transform(tokens)


if __name__ == "__main__":
    lemmatizer = Lemmatizer()
    print(lemmatizer.transform(["Customers", "bought", "deliveries", "."]))
