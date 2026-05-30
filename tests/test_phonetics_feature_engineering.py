from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


FEATURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "_01_NLP"
    / "_02_feature_engineering"
)


def load_feature_module(filename: str) -> ModuleType:
    path = FEATURE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class FakeWordToken:
    def __init__(self, word: str, is_alpha: bool = True) -> None:
        self.word = word
        self.is_alpha = is_alpha


class FakeTokenizedText:
    text = "late delivery."
    language = "en"
    word_tokens = [
        FakeWordToken("late"),
        FakeWordToken("delivery"),
        FakeWordToken(".", is_alpha=False),
    ]


def test_phonetics_extracts_per_word_features_from_tokenized_text() -> None:
    pytest.importorskip("jellyfish")
    module = load_feature_module("_01_phonetics.py")
    phonetics = module.Phonetics()

    result = phonetics(FakeTokenizedText())

    assert result.text == "late delivery."
    assert result.language == "en"
    assert [token.word for token in result.tokens] == ["late", "delivery"]
    assert all(token.soundex for token in result.tokens)
    assert all(token.metaphone for token in result.tokens)
    assert all(token.nysiis for token in result.tokens)
