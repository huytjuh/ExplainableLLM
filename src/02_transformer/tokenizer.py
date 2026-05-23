"""Small tokenizer used for explanation and tests.

This is intentionally not a production BPE implementation. It keeps the
mechanics visible: normalize text, split into tokens, map tokens to ids, and
decode ids back to text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class SimpleTokenizer:
    """A tiny word-and-punctuation tokenizer with explicit special tokens."""

    lowercase: bool = True
    special_tokens: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>")
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_to_id:
            for token in self.special_tokens:
                self._add_token(token)

    def normalize(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(self.normalize(text))

    def fit(self, texts: list[str], min_frequency: int = 1) -> "SimpleTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))
        for token, count in sorted(counts.items()):
            if count >= min_frequency:
                self._add_token(token)
        return self

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = ["<bos>", *tokens, "<eos>"]
        unknown_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unknown_id) for token in tokens]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        special = set(self.special_tokens)
        for token_id in ids:
            token = self.id_to_token.get(token_id, "<unk>")
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        return self._join_tokens(tokens)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return self.token_to_id[token]

    @staticmethod
    def _join_tokens(tokens: list[str]) -> str:
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

