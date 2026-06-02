from __future__ import annotations 

from dataclass import dataclass, field 


@dataclass(frozen=True)
class BoWEmbedding:
    word: str
    frequency: int

@dataclass
class BoWConfig:
    min_frequency: int

class BagOfWords:
    """Simple Bag-of-Words embedding extractor."""

    def __init__(self) -> None:
        pass

    def extract(self, tokens: list[str]) -> list[BoWEmbedding]:
        """Extract BoW embeddings from a list of tokens."""
        frequency_dict: dict[str, int] = {}
        for token in tokens:
            if token.isalpha():  # Only consider alphabetic tokens
                frequency_dict[token] = frequency_dict.get(token, 0) + 1

        return [BoWEmbedding(word=word, frequency=freq) for word, freq in frequency_dict.items()]