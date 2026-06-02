from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spellchecker import SpellChecker


@dataclass(frozen=True)
class SpellingFeatures:
    word: str 

    is_misspelled: bool
    grammar: bool
    typo: bool

    word_corrected: str | None = None
    levenshtein_distance: int | None = None
    dam_levenshtein_distance: int | None = None
    jaro_winkler_similarity: float | None = None

    phonetic_soundex: str | None = None
    phonetic_metaphone: str | None = None
    phonetic_nysiis: str | None = None

@dataclass(frozen=True)
class SpellingText:
    text: str
    features: list[SpellingFeatures]

    @property
    def misspelled_words(self) -> list[str]:
        return [feature.word for feature in self.features if feature.is_misspelled]
    
    @property
    def misspelled_count(self) -> int:
        return sum(feature.is_misspelled for feature in self.features)
    
    @property
    def misspelled_ratio(self) -> float:
        return self.misspelled_count / len(self.features) if self.features else 0.0
    
    @property
    def grammar_ratio(self) -> float:
        return self.grammar_count / len(self.features) if self.features else 0.0

    @property
    def typo_ratio(self) -> float:
        return self.typo_count / len(self.features) if self.features else 0.0
    
@dataclass
class SpellingConfig:
    pyspellchecker: bool = True

    levenshtein_distance: bool = True
    dam_levenshtein_distance: bool = True
    jaro_winkler_similarity: bool = True
    phonetic_soundex: bool = True
    phonetic_metaphone: bool = True
    phonetic_nysiis: bool = True


class Spelling:
    """Extract Dutch/English spelling features from tokenized text."""

    def __init__(self, config: SpellingConfig | None = None) -> None:
        """Initialize the Spelling extractor with optional configuration."""
        self.config = config or SpellingConfig()
        self.spellchecker: dict[str, SpellChecker] = {}

    def extract_word(self, word: str, language: str) -> SpellingFeatures:
        """Extract spelling features for a single word."""
        if self.spellchecker.get(language) is None:
            self.spellchecker[language] = SpellChecker(language=language)

        is_misspelled = bool(word in self.spellchecker[language])
        word_corrected = self.spellchecker[language].correction(word)

        lev_distance = self._levenshtein_distance(word, word_corrected) if self.config.levenshtein_distance else None
        dam_lev_distance = self._dam_levenshtein_distance(word, word_corrected) if self.config.dam_levenshtein_distance else None
        jaro_winkler_sim = self._jaro_winkler_similarity(word, word_corrected) if self.config.jaro_winkler_similarity else None
        phonetic_soundex = self._phonetic_soundex(word, word_corrected) if self.config.phonetic_soundex else None
        phonetic_metaphone = self._phonetic_metaphone(word, word_corrected) if self.config.phonetic_metaphone else None
        phonetic_nysiis = self._phonetic_nysiis(word, word_corrected) if self.config.phonetic_nysiis else None

        grammar = False  # Replace with actual grammar check
        typo = False  # Replace with actual typo check

        return SpellingFeatures(
            word=word,
            is_misspelled=is_misspelled,
            grammar=grammar,
            typo=typo,
            word_corrected=word_corrected,
            levenshtein_distance=lev_distance,
            dam_levenshtein_distance=dam_lev_distance,
            jaro_winkler_similarity=jaro_winkler_sim,
            phonetic_soundex=phonetic_soundex,
            phonetic_metaphone=phonetic_metaphone,
            phonetic_nysiis=phonetic_nysiis,
        )
    
    def extract(self, tokenized: Any) -> SpellingText:
        """Extract spelling features from tokenized text."""
        list_words, list_language = tokenized.get_words, tokenized.get_languages
        spelling_features = [self.extract_word(word, language) for word, language in zip(list_words, list_language)]
        return SpellingText(
            text=tokenized.text,
            features=spelling_features
        )
    
    def _levenshtein_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein distance between two words."""

        return
    
    def _dam_levenshtein_distance(self, word1: str, word2: str) -> int:
        """Calculate Damerau-Levenshtein distance between two words."""
        return
    
    def _jaro_winkler_similarity(self, word1: str, word2: str) -> float:
        """Calculate Jaro-Winkler similarity between two words."""
        return
    
    def _phonetic_soundex(self, word: str) -> str:
        """Calculate Soundex phonetic encoding for a word."""
        return

    def _phonetic_metaphone(self, word: str) -> str:
        """Calculate Metaphone phonetic encoding for a word."""
        return
    
    def _phonetic_nysiis(self, word: str) -> str:
        """Calculate NYSIIS phonetic encoding for a word."""
        return

    def __call__(self, tokenized: Any) -> SpellingText:
        """Allow the Spelling extractor to be called directly on tokenized text."""
        return self.extract(tokenized)
    
