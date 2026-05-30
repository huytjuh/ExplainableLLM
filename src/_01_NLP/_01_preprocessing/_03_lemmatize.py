from __future__ import annotations

from dataclasses import dataclass

from src._01_NLP._01_preprocessing._01_tokenize import Tokenizer


@dataclass(frozen=True)
class LemmatizedText:
    text: str
    lemmatized_text: str


class Lemmatizer:
    """Lemmatize text using spaCy token lemmas."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        """Initialize the Lemmatizer with a Tokenizer instance."""
        self.tokenizer = tokenizer

    def lemmatize(self, text: str) -> LemmatizedText:
        """Lemmatize the input text by replacing words with their lemmas when available."""
        tokenized = self.tokenizer(text)
        lemmatized_tokens = [
            token.lemma if token.lemma else token.word
            for token in tokenized.word_tokens
        ]
        lemmatized_text = " ".join(lemmatized_tokens)

        return LemmatizedText(text=text, lemmatized_text=lemmatized_text)

    def __call__(self, text: str) -> LemmatizedText:
        """Allow the Lemmatizer to be called directly on text."""
        return self.lemmatize(text)