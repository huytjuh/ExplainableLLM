"""Protect important terms during preprocessing.

Some terms should survive normalization, tokenization, stemming, or stopword
removal as one unit. Examples include product names, model names, legal terms,
email addresses, URLs, or domain-specific phrases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


DEFAULT_PATTERNS = {
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "url": r"https?://[^\s]+",
}


@dataclass(frozen=True)
class ProtectedTerms:
    """Mask protected terms before preprocessing and restore them afterwards."""

    terms: tuple[str, ...] = ()
    patterns: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PATTERNS))
    placeholder_prefix: str = "__PROTECTED"

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace protected spans with placeholders.

        Returns the protected text and a mapping from placeholder to original
        value. The mapping is passed to :meth:`restore` after preprocessing.
        """
        protected_text = text
        replacements: dict[str, str] = {}

        for match in self._find_matches(text):
            placeholder = f"{self.placeholder_prefix}_{len(replacements)}__"
            replacements[placeholder] = match.group(0)
            protected_text = protected_text.replace(match.group(0), placeholder, 1)

        return protected_text, replacements

    def restore(self, text: str, replacements: dict[str, str]) -> str:
        """Restore placeholders in a processed string."""
        restored = text
        for placeholder, original in replacements.items():
            restored = restored.replace(placeholder, original)
            restored = restored.replace(placeholder.lower(), original)
        return restored

    def restore_tokens(
        self,
        tokens: list[str],
        replacements: dict[str, str],
    ) -> list[str]:
        """Restore placeholders in a processed token sequence."""
        restored = []
        lower_replacements = {
            placeholder.lower(): original
            for placeholder, original in replacements.items()
        }

        for token in tokens:
            restored.append(replacements.get(token, lower_replacements.get(token, token)))

        return restored

    def _find_matches(self, text: str) -> list[re.Match[str]]:
        matches: list[re.Match[str]] = []

        for term in self.terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
            matches.extend(pattern.finditer(text))

        for pattern in self.patterns.values():
            matches.extend(re.finditer(pattern, text))

        return sorted(matches, key=lambda match: match.start())


if __name__ == "__main__":
    protector = ProtectedTerms(terms=("Gemini 2.5 Flash Lite", "New York"))
    masked, lookup = protector.protect("Email Gemini 2.5 Flash Lite at llm@example.com.")
    print(masked)
    print(protector.restore(masked.lower(), lookup))
