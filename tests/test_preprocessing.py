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


def test_text_normalizer_cleans_common_raw_text_noise() -> None:
    module = load_module("_00_normalize.py")
    normalizer = module.TextNormalizer(strip_accents=True)

    assert normalizer.normalize("  Café\tCUSTOMER\x00feedback\n ") == (
        "cafe customer feedback"
    )


def test_tokenizer_splits_words_contractions_and_punctuation() -> None:
    module = load_module("_01_tokenize.py")
    tokenizer = module.Tokenizer()

    assert tokenizer.tokenize("Customers can't log in.") == [
        "customers",
        "can't",
        "log",
        "in",
        ".",
    ]


def test_spacy_language_tokenizer_classifies_parts_and_uses_matching_pipeline() -> None:
    module = load_module("_01_tokenize.py")

    class FakeAttrs:
        lang: str | None = None

    class FakeToken:
        def __init__(self, text: str, lang: str | None = None) -> None:
            self.text = text
            self.is_punct = text in {".", "!", "?"}
            self.is_space = False
            self._ = FakeAttrs()
            self._.lang = lang

    class FakeBasePipeline:
        def __call__(self, text: str) -> list[FakeToken]:
            return [FakeToken(piece) for piece in text.replace(".", " .").split()]

    class FakeLangPipeline:
        labels = {
            "The": "EN",
            "delivery": "EN",
            "is": "EN",
            "late": "EN",
            "De": "NL",
            "levering": "NL",
            "kwam": "NL",
            "te": "NL",
            "laat": "NL",
            ".": "PUNCT",
        }

        def __call__(self, doc: list[FakeToken]) -> list[FakeToken]:
            for token in doc:
                token._.lang = self.labels[token.text]
            return doc

    class FakePipeline:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def __call__(self, text: str) -> list[FakeToken]:
            pieces = text.split()
            return [FakeToken(f"{self.prefix}:{piece}") for piece in pieces]

    tokenizer = module.SpacyLanguageTokenizer(
        base_pipeline=FakeBasePipeline(),
        lang_pipeline=FakeLangPipeline(),
        english_pipeline=FakePipeline("en"),
        dutch_pipeline=FakePipeline("nl"),
    )

    parts = tokenizer.tokenize_parts("The delivery is late. De levering kwam te laat.")

    assert [part.language for part in parts] == ["EN", "NL"]
    assert parts[0].tokens == ["en:the", "en:delivery", "en:is", "en:late"]
    assert parts[1].tokens == ["nl:de", "nl:levering", "nl:kwam", "nl:te", "nl:laat"]


def test_rule_based_stemmer_keeps_punctuation() -> None:
    module = load_module("_02_stemming.py")
    stemmer = module.RuleBasedStemmer()

    assert stemmer.transform(["Customers", "deliveries", "."]) == [
        "custom",
        "delivery",
        ".",
    ]


def test_lemmatizer_prefers_lexicon_before_fallback_rules() -> None:
    module = load_module("_03_lemmatize.py")
    lemmatizer = module.Lemmatizer()

    assert lemmatizer.transform(["Customers", "bought", "orders"]) == [
        "customer",
        "buy",
        "order",
    ]


def test_stopword_remover_can_preserve_negations() -> None:
    module = load_module("_04_stopwords.py")
    remover = module.StopwordRemover(language="both")

    assert remover.remove(["the", "delivery", "is", "not", "on", "time"]) == [
        "delivery",
        "not",
        "time",
    ]


def test_protected_terms_mask_and_restore_domain_terms() -> None:
    module = load_module("_05_protect_terms.py")
    protector = module.ProtectedTerms(terms=("Gemini 2.5 Flash Lite",))

    masked, replacements = protector.protect(
        "Email Gemini 2.5 Flash Lite at llm@example.com."
    )

    assert masked == "Email __PROTECTED_0__ at __PROTECTED_1__."
    assert protector.restore(masked.lower(), replacements) == (
        "email Gemini 2.5 Flash Lite at llm@example.com."
    )
