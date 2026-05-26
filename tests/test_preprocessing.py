from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PREPROCESSING_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "01_NLP" / "01_preprocessing"
)


def load_module(filename: str) -> ModuleType:
    path = PREPROCESSING_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_tokenizer_splits_words_contractions_and_punctuation() -> None:
    module = load_module("01_tokenize.py")
    tokenizer = module.Tokenizer()

    assert tokenizer.tokenize("Customers can't log in.") == [
        "customers",
        "can't",
        "log",
        "in",
        ".",
    ]


def test_rule_based_stemmer_keeps_punctuation() -> None:
    module = load_module("02_stemming.py")
    stemmer = module.RuleBasedStemmer()

    assert stemmer.transform(["Customers", "deliveries", "."]) == [
        "custom",
        "delivery",
        ".",
    ]


def test_lemmatizer_prefers_lexicon_before_fallback_rules() -> None:
    module = load_module("03_lemmatize.py")
    lemmatizer = module.Lemmatizer()

    assert lemmatizer.transform(["Customers", "bought", "orders"]) == [
        "customer",
        "buy",
        "order",
    ]


def test_stopword_remover_can_preserve_negations() -> None:
    module = load_module("04_stopwords.py")
    remover = module.StopwordRemover(language="both")

    assert remover.remove(["the", "delivery", "is", "not", "on", "time"]) == [
        "delivery",
        "not",
        "time",
    ]


def test_protected_terms_mask_and_restore_domain_terms() -> None:
    module = load_module("05_protect_terms.py")
    protector = module.ProtectedTerms(terms=("Gemini 2.5 Flash Lite",))

    masked, replacements = protector.protect(
        "Email Gemini 2.5 Flash Lite at llm@example.com."
    )

    assert masked == "Email __PROTECTED_0__ at __PROTECTED_1__."
    assert protector.restore(masked.lower(), replacements) == (
        "email Gemini 2.5 Flash Lite at llm@example.com."
    )
