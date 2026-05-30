from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jellyfish


@dataclass(frozen=True)
class PhoneticFeatures:
    word: str
    soundex: str | None
    metaphone: str | None
    nysiis: str | None

@dataclass(frozen=True)
class PhoneticText:
    text: str
    features: list[PhoneticFeatures]

    @property
    def get_soundexes(self) -> list[str]:
        return [feature.soundex for feature in self.features if feature.soundex]

    @property
    def get_metaphones(self) -> list[str]:
        return [feature.metaphone for feature in self.features if feature.metaphone]

    @property
    def get_nysiises(self) -> list[str]:
        return [feature.nysiis for feature in self.features if feature.nysiis]

@dataclass
class PhoneticConfig:
    soundex: bool = True
    metaphone: bool = True
    nysiis: bool = True
    alpha_only: bool = True


class Phonetics:
    """Extract phonetic features using Soundex, Metaphone, and NYSIIS."""

    def __init__(self, config: PhoneticConfig | None = None) -> None:
        """Initialize the Phonetics extractor with the given configuration."""
        self.config = config or PhoneticConfig()

    def extract_word(self, word: str) -> PhoneticFeatures:
        """Extract phonetic features for a single word."""
        soundex = self._soundex(word) if self.config.soundex else None
        metaphone = self._metaphone(word) if self.config.metaphone else None
        nysiis = self._nysiis(word) if self.config.nysiis else None

        return PhoneticFeatures(
            word=word,
            soundex=soundex,
            metaphone=metaphone,
            nysiis=nysiis,
        )

    def extract(self, tokenized: Any) -> PhoneticText:
        """Extract phonetic features from tokenized text."""
        phonetic_features = [self.extract_word(word) for word in tokenized.get_words]

        return PhoneticText(
            text=tokenized.text,
            features=phonetic_features
        )

    def _soundex(self, text: str) -> str:
        """Compute the Soundex code for the given text."""
        return jellyfish.soundex(text)

    def _metaphone(self, text: str) -> str:
        """Compute the Metaphone code for the given text."""
        return jellyfish.metaphone(text)

    def _nysiis(self, text: str) -> str:
        """Compute the NYSIIS code for the given text."""
        return jellyfish.nysiis(text)

    def __call__(self, value: str | Any) -> PhoneticText:
        """Allow the Phonetics extractor to be called directly on text or tokenized input."""
        return self.extract(value)
