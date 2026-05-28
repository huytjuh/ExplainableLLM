"""Tokenization examples for classic NLP preprocessing.

Tokenization is the step that turns raw text into a sequence of smaller units.
This example keeps punctuation as separate tokens so later preprocessing steps
can decide whether to keep or drop it.
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.util import compile_infix_regex


@dataclass(frozen=True)
class TokenizedText:
    text: str
    tokens: list[str]


@dataclass 
class TokenizerConfig:
    model: str|None=None
    sentencizer: bool=True
    keep_hyphens: bool=True


class Tokenizer:
    """ SpaCy-based tokenizer that detects English and Dutch spans and tokenizes them with the appropriate model. """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Initialize the tokenizer with the specified language model."""
        config = config or TokenizerConfig()
        self.nlp = spacy.load(config.model) if config.model else spacy.blank("xx")

        if config.sentencizer:
            self.nlp.add_pipe("sentencizer")

        if config.keep_hyphens:
            infixes = [x for x in self.nlp.Defaults.infixes if "-" not in x]
            self.nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
        
    def tokenize(self, text: str) -> list[str]:
        """Tokenize the text into a list of tokens."""
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]


if __name__ == "__main__":

    text = "Ik cant login bij ING-app, keeps crashen btw. OpenAI ChatGPT-5 werkt asap."

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    print(tokens)    
    
