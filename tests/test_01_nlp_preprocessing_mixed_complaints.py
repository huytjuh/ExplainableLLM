from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


PREPROCESSING_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "_01_NLP"
    / "_01_preprocessing"
)


def load_preprocessing_module(filename: str) -> ModuleType:
    path = PREPROCESSING_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


MIXED_COMPLAINT_DESCRIPTIONS = [
    "The delivery was late en de klantenservice reageerde niet!!!",
    "Ik kan niet inloggen; checkout keeps crashing at payment.",
    "Refund was not processed en de factuur klopt niet.",
]


def test_normalizer_cleans_mixed_dutch_english_complaint_descriptions() -> None:
    normalize_module = load_preprocessing_module("_00_normalize.py")
    normalizer_class = normalize_module.Normalizer
    normalizer = normalizer_class()

    normalized = [
        normalizer.normalize(description).text
        for description in MIXED_COMPLAINT_DESCRIPTIONS
    ]

    assert normalized == [
        "the delivery was late en de klantenservice reageerde niet.",
        "ik kan niet inloggen. checkout keeps crashing at payment.",
        "refund was not processed en de factuur klopt niet.",
    ]


def test_stopword_remover_keeps_complaint_signal_and_negations() -> None:
    stopwords_module = load_preprocessing_module("_04_stopwords.py")
    remover_class = stopwords_module.StopwordRemover
    remover = remover_class(language="both")

    tokens = [
        "the",
        "delivery",
        "was",
        "late",
        "en",
        "de",
        "klantenservice",
        "reageerde",
        "niet",
        "not",
    ]

    assert remover.remove(tokens) == [
        "delivery",
        "late",
        "klantenservice",
        "reageerde",
        "niet",
        "not",
    ]


def test_stopword_remover_can_remove_negations_when_configured() -> None:
    stopwords_module = load_preprocessing_module("_04_stopwords.py")
    remover_class = stopwords_module.StopwordRemover
    remover = remover_class(
        language="both",
        extra_stopwords={"not", "niet", "geen"},
        keep_negations=False,
    )

    assert remover.remove(["not", "niet", "geen", "delivery", "factuur"]) == [
        "delivery",
        "factuur",
    ]


def test_tokenizer_handles_mixed_complaint_sentences_without_language_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("spacy")
    pytest.importorskip("langdetect")

    tokenize_module = load_preprocessing_module("_01_tokenize.py")
    tokenizer_class = tokenize_module.Tokenizer
    tokenizer_config_class = tokenize_module.TokenizerConfig

    tokenizer = tokenizer_class(tokenizer_config_class(english=False, dutch=False))

    def detect_language(text: str) -> str:
        dutch_markers = {"de", "en", "ik", "inloggen", "klantenservice", "niet"}
        words = {piece.lower().strip(".") for piece in text.split()}
        return "nl" if words & dutch_markers else "en"

    monkeypatch.setattr(tokenizer, "_detect_language", detect_language)

    text = " ".join(
        load_preprocessing_module("_00_normalize.py").Normalizer()
        .normalize(description)
        .text
        for description in MIXED_COMPLAINT_DESCRIPTIONS
    )

    tokenized = tokenizer.tokenize(text)
    words = [token.word.lower() for token in tokenized.word_tokens]

    assert tokenized.language == "nl"
    assert [sentence.language for sentence in tokenized.sent_tokens] == [
        "nl",
        "nl",
        "en",
        "nl",
    ]
    assert {"delivery", "klantenservice", "inloggen", "checkout", "payment"}.issubset(
        words
    )
    assert all(token.lemma is None for token in tokenized.word_tokens)
