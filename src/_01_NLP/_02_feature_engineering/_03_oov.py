from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class OOVFeatures:
    word: str

@dataclass(frozen=True)
class OOVText:
    text: str
    features: list[OOVFeatures]

@dataclass
class OOVConfig:
    languages: tuple[str, ...] = ("en", "nl")


class OutOfVocabulary:
    """Extract Dutch/English out-of-vocabulary features from tokenized text."""

    def __init__(self, config: OOVConfig | None = None) -> None:
        self.config = config or OOVConfig()

    def extract_word(self, word: str, language: str | None = None) -> OOVFeatures:
        normalized = self._normalize(word)
        known, source = self._lookup(normalized, language)

        return OOVFeatures(
            word=word,
            normalized=normalized,
            language=language,
            in_vocabulary=known,
            oov=not known,
            source=source,
        )

    def extract(self, tokenized: Any) -> OOVText:
        language = getattr(tokenized, "language", None)
        features = [
            self.extract_word(token.word, language=language)
            for token in tokenized.word_tokens
            if self._should_extract(token)
        ]

        return OOVText(
            text=getattr(tokenized, "text", ""),
            features=features,
        )

    def _build_vocabulary_by_language(self) -> dict[str, set[str]]:
        vocabularies: dict[str, set[str]] = {}

        if self.config.use_basic_vocab:
            vocabularies["en"] = set(BASIC_ENGLISH_VOCAB)
            vocabularies["nl"] = set(BASIC_DUTCH_VOCAB)

        for language, vocabulary in self.config.vocabulary_by_language.items():
            vocabularies.setdefault(language, set()).update(
                self._normalize(word) for word in vocabulary
            )

        extra_vocabulary = {self._normalize(word) for word in self.config.extra_vocabulary}
        for language in self.config.languages:
            vocabularies.setdefault(language, set()).update(extra_vocabulary)

        return vocabularies

    def _lookup(self, word: str, language: str | None) -> tuple[bool, str | None]:
        candidate_languages = self._candidate_languages(word, language)

        for candidate_language in candidate_languages:
            vocabulary = self._vocabulary_by_language.get(candidate_language, set())
            if word in vocabulary:
                return True, f"{candidate_language}_vocabulary"

        if self.config.use_spacy_vectors:
            for candidate_language in candidate_languages:
                if self._spacy_knows(word, candidate_language):
                    return True, f"{candidate_language}_spacy_vectors"

        return False, None

    def _candidate_languages(self, word: str, language: str | None) -> list[str]:
        if language in self.config.languages:
            return [language, *[item for item in self.config.languages if item != language]]

        direct_matches = [
            language
            for language, vocabulary in self._vocabulary_by_language.items()
            if word in vocabulary
        ]
        if direct_matches:
            return direct_matches

        return list(self.config.languages)

    def _spacy_knows(self, word: str, language: str) -> bool:
        nlp = self._get_spacy_pipeline(language)
        if nlp is None:
            return False

        lexeme = nlp.vocab[word]
        return bool(lexeme.has_vector and not lexeme.is_oov)

    def _get_spacy_pipeline(self, language: str) -> Any | None:
        if language in self._spacy_pipelines:
            return self._spacy_pipelines[language]

        model_name = self.config.spacy_models.get(language)
        if model_name is None:
            self._spacy_pipelines[language] = None
            return None

        try:
            import spacy

            self._spacy_pipelines[language] = spacy.load(model_name)
        except (ImportError, OSError):
            self._spacy_pipelines[language] = None

        return self._spacy_pipelines[language]

    def _should_extract(self, token: Any) -> bool:
        if not self.config.alpha_only:
            return True
        return bool(getattr(token, "is_alpha", str(token.word).isalpha()))

    def _normalize(self, word: str) -> str:
        word = word.strip()
        if self.config.lowercase:
            word = word.lower()
        return word

    def __call__(self, tokenized: Any) -> OOVText:
        return self.extract(tokenized)
