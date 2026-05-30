from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import spacy
from spacy.tokens import Token
from spacy.util import compile_infix_regex
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


DetectorFactory.seed = 42

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPACY_EN = PROJECT_ROOT / "models/spacy/en_core_web_sm-3.8.0"
SPACY_NL = PROJECT_ROOT / "models/spacy/nl_core_news_sm-3.8.0"


@dataclass(frozen=True)
class WordToken:
    word: str
    is_alpha: bool
    is_stop: bool
    lemma: str | None = None
    pos: str | None = None
    tag: str | None = None
    dep: str | None = None
    shape: str | None = None


@dataclass(frozen=True)
class SentenceToken:
    sentence: str
    language: str
    tokens: list[WordToken]


@dataclass(frozen=True)
class TokenizedText:
    text: str
    language: str
    word_tokens: list[WordToken]
    sent_tokens: list[SentenceToken]

    @property
    def get_words(self) -> list[str]:
        return [token.word for token in self.word_tokens if token.is_alpha]
    
    @property
    def get_sentences(self) -> list[str]:
        return [sent.sentence for sent in self.sent_tokens]


@dataclass
class TokenizerConfig:
    english: bool = True
    dutch: bool = True
    sentencizer: bool = True
    keep_hyphens: bool = True


class Tokenizer:
    """Two-stage tokenizer with optional English and Dutch spaCy enrichment."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        """Two-stage tokenizer with optional English and Dutch spaCy enrichment."""
        self.config = config or TokenizerConfig()
        self.nlp = spacy.blank("xx")

        self.nlp_en = spacy.load(str(SPACY_EN)) if self.config.english else None
        self.nlp_nl = spacy.load(str(SPACY_NL)) if self.config.dutch else None

        if self.config.sentencizer:
            self.nlp.add_pipe("sentencizer")

        if self.config.keep_hyphens:
            infixes = [x for x in self.nlp.Defaults.infixes if x != "-"]
            self.nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer

    def simple_tokenize(self, text: str) -> tuple[list[str], list[WordToken]]:
        """Basic tokenization into sentences and word tokens without linguistic features."""
        doc = self.nlp(text)

        sentences = [sent.text.strip() for sent in doc.sents]
        tokens = [
            WordToken(
                word=token.text,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
            )
            for token in doc
            if not token.is_space
        ]

        return sentences, tokens

    def linguistic_tokenize(self, text: str, language: str) -> SentenceToken:
        """Linguistically enriched tokenization using spaCy models for English and Dutch."""
        if language == "en" and self.nlp_en is not None:
            doc = self.nlp_en(text)
        elif language == "nl" and self.nlp_nl is not None:
            doc = self.nlp_nl(text)
        else:
            _, fallback_tokens = self.simple_tokenize(text)
            return SentenceToken(sentence=text, language=language, tokens=fallback_tokens)

        tokens = [
            WordToken(
                word=token.text,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
                lemma=token.lemma_,
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                shape=token.shape_,
            )
            for token in doc
            if not token.is_space
        ]

        return SentenceToken(sentence=text, language=language, tokens=tokens)

    def tokenize(self, text: str) -> TokenizedText:
        """Tokenize the input text into sentences and word tokens, with language detection and optional linguistic enrichment."""
        sentences, _ = self.simple_tokenize(text)
        doc_language = self._detect_language(text)

        sent_tokens: list[SentenceToken] = []
        for sent in sentences:
            language = self._detect_language(sent)
            sent_tokens.append(self.linguistic_tokenize(sent, language))

        word_tokens = [token for sent in sent_tokens for token in sent.tokens]

        return TokenizedText(
            text=text,
            language=doc_language,
            word_tokens=word_tokens,
            sent_tokens=sent_tokens,
        )

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text using langdetect, with fallback to Dutch."""
        try:
            language = detect(text).split("-")[0]
        except LangDetectException:
            return "nl"

        if language not in ["en", "nl"]:
            return 'nl'

        return language

    def __call__(self, text: str) -> TokenizedText:
        """Allow the Tokenizer to be called directly on text."""
        return self.tokenize(text)