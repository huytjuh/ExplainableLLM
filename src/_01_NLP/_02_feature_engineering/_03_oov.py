from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import spacy
from spacy.tokens import Token

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPACY_EN = PROJECT_ROOT / "models/spacy/en_core_web_md-3.8.0"
SPACY_NL = PROJECT_ROOT / "models/spacy/nl_core_news_md-3.8.0"

@dataclass(frozen=True)
class OOVFeatures:
    word: str
    oov: bool

@dataclass(frozen=True)
class OOVText:
    text: str
    features: list[OOVFeatures]

    @property
    def oov_words(self) -> list[str]:
        return [feature.word for feature in self.features if feature.oov]
    
    @property
    def oov_count(self) -> int:
        return sum(feature.oov for feature in self.features)
    
    @property
    def oov_ratio(self) -> float:
        return self.oov_count / len(self.features) if self.features else 0.0

@dataclass
class OOVConfig:
    spacy_models: dict[str, Path] = field(
    default_factory=lambda: {
        'en': SPACY_EN,
        'nl': SPACY_NL,
        }
    )


class OutOfVocabulary:
    """Extract Dutch/English out-of-vocabulary features from tokenized text."""

    def __init__(self, config: OOVConfig | None = None) -> None:
        """Initialize the OutOfVocabulary extractor with optional configuration."""
        self.config = config or OOVConfig()
        self.nlp_models: dict[str, Any | None] = {}

    def extract_word(self, word: str, language: str | None = None) -> OOVFeatures:
        """Extract OOV features for a single word, optionally using language information."""
        if self.nlp_models.get(language) is None and language in self.config.spacy_models:
            self.nlp_models[language] = spacy.load(self.config.spacy_models[language])

        nlp = self.nlp_models[language]
        if nlp:
            lexeme = nlp.vocab[word]
            oov = bool(lexeme.is_oov)
 
        return OOVFeatures(
            word=word,
            oov=oov if nlp else True
        )
    
    def extract(self, tokenized: Any) -> OOVText:
        """Extract OOV features from tokenized text, using language information if available."""
        list_words, list_language = tokenized.get_words, tokenized.get_languages
        oov_features = [self.extract_word(word, language) for word, language in zip(list_words, list_language)]
        return OOVText(
            text=tokenized.text,
            features=oov_features
        )
     
    def __call__(self, value: str | Any) -> OOVText:
        """Allow the OutOfVocabulary extractor to be called directly on text or tokenized input."""
        return self.extract(value)