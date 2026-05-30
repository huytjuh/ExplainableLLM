from __future__ import annotations

from dataclasses import dataclass

from src._01_NLP._01_preprocessing._01_tokenize import Tokenizer


@dataclass(frozen=True)
class StemmedText:
    text: str
    stemmed_text: str


class Stemmer:
    """Stem text using available token lemmas."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Initialize the Stemmer with a Tokenizer instance."""
        self.tokenizer = tokenizer

    def stem(self, text: str) -> StemmedText:
        """Stem the input text by replacing words with their lemmas when available."""
        tokenized = self.tokenizer(text)
        stemmed_tokens = [
            token.lemma if token.lemma else token.word
            for token in tokenized.word_tokens
        ]
        stemmed_text = " ".join(stemmed_tokens)

        return StemmedText(text=text, stemmed_text=stemmed_text)

    def __call__(self, text: str) -> StemmedText:
        """Allow the Stemmer to be called directly on text."""
        return self.stem(text)