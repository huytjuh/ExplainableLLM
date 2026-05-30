from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LexicalFeatures:
    word: str

    length: int
    is_stop: bool
    is_upper: bool
    is_digit: bool 

@dataclass(frozen=True)
class LexicalText:
    text: str
    features: list[LexicalFeatures]

    @property
    def average_word_length(self) -> float:
        return sum(feature.length for feature in self.features) / len(self.features)
    
    @property 
    def stop_word_ratio(self) -> float:
        return sum(feature.is_stop for feature in self.features) / len(self.features)
    
    @property
    def uppercase_ratio(self) -> float:
        return sum(feature.is_upper for feature in self.features) / len(self.features)
    
    @property
    def digit_ratio(self) -> float:
        return sum(feature.is_digit for feature in self.features) / len(self.features)

@dataclass
class LexicalConfig:
    basic: bool = True


class Lexical:
    """Extract lexical features such as word length, alphabetic status, stop word status, and uppercase status."""

    def __init__(self, config: LexicalConfig | None = None) -> None:
        """Initialize the Lexical extractor with the given configuration."""
        self.config = config or LexicalConfig()

    def extract_word(self, word: str) -> LexicalFeatures:
        """Extract lexical features for a single word."""
        return LexicalFeatures(
            word=word,
            length=len(word),
            is_stop=word.lower() in {"the", "is", "in", "and", "to", "a"},  # Simple stop word list
            is_upper=word.isupper(),
            is_digit=word.isdigit()
        )
    
    def extract(self, tokenized: Any) -> LexicalText:
        """Extract lexical features from tokenized text."""
        lexical_features = [self.extract_word(word) for word in tokenized.get_words]
        return LexicalText(
            text=tokenized.text,
            features=lexical_features
        )

    def __call__(self, value: str | Any) -> LexicalText:
        """Allow the Lexical extractor to be called directly on text or tokenized input."""
        return self.extract(value)