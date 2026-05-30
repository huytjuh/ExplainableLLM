from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
    text = "refund was not processed en de factuur klopt niet delivry."
    language = "nl"
    word_tokens = [
        FakeWordToken("refund"),
        FakeWordToken("was"),
        FakeWordToken("not"),
        FakeWordToken("processed"),
        FakeWordToken("en"),
        FakeWordToken("de"),
        FakeWordToken("factuur"),
        FakeWordToken("klopt"),
        FakeWordToken("niet"),
        FakeWordToken("delivry"),
        FakeWordToken(".", is_alpha=False),
    ]


def test_oov_extracts_known_and_unknown_dutch_english_complaint_words() -> None:
    module = load_feature_module("_03_oov.py")
    extractor = module.OutOfVocabulary()

    result = extractor(FakeTokenizedText())

    by_word = {feature.word: feature for feature in result.features}

    assert by_word["refund"].oov is False
    assert by_word["factuur"].oov is False
    assert by_word["delivry"].oov is True
    assert result.oov_count == 1
    assert result.token_count == 10
    assert result.oov_ratio == 0.1


def test_oov_accepts_extra_domain_vocabulary() -> None:
    module = load_feature_module("_03_oov.py")
    config = module.OOVConfig(extra_vocabulary={"delivry"})
    extractor = module.OutOfVocabulary(config)

    result = extractor(FakeTokenizedText())

    assert result.oov_count == 0
