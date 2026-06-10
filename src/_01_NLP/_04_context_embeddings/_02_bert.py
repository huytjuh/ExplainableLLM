"""Trainable encoder-only BERT-style embeddings with PyTorch.

This is a compact educational implementation. It builds a vocabulary, creates
token + position + segment embeddings, passes them through bidirectional
Transformer encoder layers, and trains with a masked language modeling
objective like BERT.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class BERTEmbedding:
    token: str
    token_id: int

    context_embedding: Vector
    word_embedding: Vector
    position_embedding: Vector
    segment_embedding: Vector
    self_attention: Vector


@dataclass(frozen=True)
class BERTEncoderOutput:
    tokens: list[str]
    token_ids: list[int]
    token_embeddings: list[BERTEmbedding]
    sequence_embedding: Vector
    attention_weights: Matrix


@dataclass
class BERTConfig:
    embedding_dim: int = 16
    max_position_embeddings: int = 128
    num_encoder_layers: int = 1
    num_attention_heads: int = 2
    intermediate_dim: int = 32
    dropout: float = 0.1
    seed: int = 13
    lowercase: bool = True
    add_special_tokens: bool = True
    mask_probability: float = 0.15
    epochs: int = 25
    batch_size: int = 8
    learning_rate: float = 1e-3
    device: str = "cpu"

    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    mask_token: str = "[MASK]"

class BERTEncoder(nn.Module):
    """Trainable PyTorch encoder for BERT-style contextual embeddings."""

    def __init__(self, config: BERTConfig | None = None) -> None:
        """Initialize the trainable PyTorch encoder."""
        super().__init__()
        self.config = config or BERTConfig()



class BERT:
    """Bi-directional Encoder Representations from Transformers (BERT), a Encoder-Only model from scratch."""

    def __init__(self, config: BERTConfig | None = None) -> None:
        self.config = config or BERTConfig()

    def fit(self, corpus: list[Any]) -> 'BERT':
        """Fit a trainable BERT encoder and return the model."""
        return self
    
    def transform(self, texts: list[Any]) -> list[BERTEncoderOutput]:
        """Encode a batch of texts into contextual embeddings."""
        return

    def fit_transform(self, corpus: list[Any]) -> list[BERTEncoderOutput]:
        """Fit a trainable BERT encoder and return the encoded corpus."""
        return




class BERTEncoderLayer(nn.Module):
    """One bidirectional Transformer encoder layer."""

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_dim, config.embedding_dim),
        )
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        key_padding_mask = attention_mask == 0
        attended, attention_weights = self.self_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        hidden_states = self.attention_norm(hidden_states + self.dropout(attended))
        feed_forward = self.feed_forward(hidden_states)
        hidden_states = self.output_norm(hidden_states + self.dropout(feed_forward))
        return hidden_states, attention_weights


class BERTEncoder(nn.Module):
    """Neural network that produces BERT-style contextual token embeddings."""

    def __init__(self, config: BERTConfig, vocabulary_size: int, pad_token_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.token_embeddings = nn.Embedding(
            vocabulary_size,
            config.embedding_dim,
            padding_idx=pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.embedding_dim,
        )
        self.segment_embeddings = nn.Embedding(2, config.embedding_dim)
        self.embedding_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [BERTEncoderLayer(config) for _ in range(config.num_encoder_layers)]
        )
        self.mlm_head = nn.Linear(config.embedding_dim, vocabulary_size)

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        batch_size, sequence_length = input_ids.shape
        position_ids = torch.arange(sequence_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)

        hidden_states = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.segment_embeddings(segment_ids)
        )
        hidden_states = self.dropout(self.embedding_norm(hidden_states))

        all_attention_weights: list[Tensor] = []
        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            all_attention_weights.append(attention_weights)

        logits = self.mlm_head(hidden_states)
        return hidden_states, logits, all_attention_weights


class BERT:
    """Bidirectional Encoder Representations from Transformers from scratch."""

    def __init__(self, config: BERTConfig | None = None) -> None:
        self.config = config or BERTConfig()
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

        self.vocabulary: list[str] = []
        self.word_idx: dict[str, int] = {}
        self.id_to_word: dict[int, str] = {}
        self.vocabulary_: dict[str, int] = self.word_idx
        self.id_to_token_: dict[int, str] = self.id_to_word
        self.encoder: BERTEncoder | None = None
        self.loss_history: list[float] = []

    def fit(self, corpus: list[Any], *, epochs: int | None = None) -> "BERT":
        """Build the vocabulary, initialize the encoder, and train with MLM."""

        tokens = [token for text in corpus for token in self._tokens_from_text(text)]
        special_tokens = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.cls_token,
            self.config.sep_token,
            self.config.mask_token,
        ]
        self.vocabulary = special_tokens + sorted(set(tokens) - set(special_tokens))
        self.word_idx = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_word = {index: token for token, index in self.word_idx.items()}
        self.vocabulary_ = self.word_idx
        self.id_to_token_ = self.id_to_word
        self.encoder = self._initialize().to(self.config.device)

        if corpus:
            self._train_masked_language_model(corpus, epochs or self.config.epochs)
        return self

    def transform(self, texts: list[Any], *, pooling: str = "mean") -> list[BERTEncoderOutput]:
        """Encode a batch of texts into contextual embeddings."""

        return [self.encode(text, pooling=pooling) for text in texts]

    def fit_transform(
        self,
        corpus: list[Any],
        *,
        epochs: int | None = None,
        pooling: str = "mean",
    ) -> list[BERTEncoderOutput]:
        return self.fit(corpus, epochs=epochs).transform(corpus, pooling=pooling)

    def encode(self, text: Any, *, pooling: str = "mean") -> BERTEncoderOutput:
        """Return token-level contextual embeddings for one input."""

        self._check_is_fit()
        assert self.encoder is not None

        tokens = self._tokens_with_specials(text)
        token_ids = self._token_ids(tokens)
        segment_ids = self._segment_ids(tokens)
        attention_mask = [1] * len(token_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.config.device)
        segments = torch.tensor([segment_ids], dtype=torch.long, device=self.config.device)
        mask = torch.tensor([attention_mask], dtype=torch.long, device=self.config.device)

        self.encoder.eval()
        with torch.no_grad():
            hidden_states, _, all_attention_weights = self.encoder(input_ids, segments, mask)

        hidden = hidden_states[0].cpu()
        attention = all_attention_weights[-1][0].cpu() if all_attention_weights else torch.empty(0)
        attention_matrix = self._attention_matrix(attention)
        sequence_embedding = self._pool(tokens, hidden, pooling)
        token_embeddings = [
            BERTEmbedding(
                token=tokens[index],
                token_id=token_ids[index],
                context_embedding=hidden[index].tolist(),
                word_embedding=self.encoder.token_embeddings.weight[token_ids[index]].detach().cpu().tolist(),
                position_embedding=self.encoder.position_embeddings.weight[index].detach().cpu().tolist(),
                segment_embedding=self.encoder.segment_embeddings.weight[segment_ids[index]].detach().cpu().tolist(),
                self_attention=attention_matrix[index] if attention_matrix else [],
            )
            for index in range(len(tokens))
        ]
        return BERTEncoderOutput(
            tokens=tokens,
            token_ids=token_ids,
            token_embeddings=token_embeddings,
            sequence_embedding=sequence_embedding,
            attention_weights=attention_matrix,
        )

    def _initialize(self) -> BERTEncoder:
        """Initialize the trainable PyTorch encoder."""

        return BERTEncoder(
            self.config,
            vocabulary_size=len(self.vocabulary),
            pad_token_id=self.word_idx[self.config.pad_token],
        )

    def _train_masked_language_model(self, corpus: list[Any], epochs: int) -> None:
        assert self.encoder is not None

        examples = [self._training_example(text) for text in corpus]
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.config.learning_rate)
        self.encoder.train()
        self.loss_history = []

        for _ in range(epochs):
            random.shuffle(examples)
            epoch_losses: list[float] = []
            for start in range(0, len(examples), self.config.batch_size):
                batch = examples[start : start + self.config.batch_size]
                input_ids, segment_ids, attention_mask, labels = self._collate(batch)
                optimizer.zero_grad()
                _, logits, _ = self.encoder(input_ids, segment_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, len(self.vocabulary)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.loss_history.append(sum(epoch_losses) / len(epoch_losses))

    def _training_example(self, text: Any) -> tuple[list[int], list[int], list[int]]:
        tokens = self._tokens_with_specials(text)
        token_ids = self._token_ids(tokens)
        input_ids = token_ids.copy()
        labels = [-100] * len(token_ids)
        mask_id = self.word_idx[self.config.mask_token]
        candidate_positions = [
            index
            for index, token in enumerate(tokens)
            if token not in {self.config.cls_token, self.config.sep_token}
        ]

        selected = [
            index
            for index in candidate_positions
            if random.random() < self.config.mask_probability
        ]
        if not selected and candidate_positions:
            selected = [random.choice(candidate_positions)]

        for index in selected:
            labels[index] = token_ids[index]
            input_ids[index] = mask_id

        return input_ids, self._segment_ids(tokens), labels

    def _collate(
        self,
        examples: list[tuple[list[int], list[int], list[int]]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        max_length = max(len(input_ids) for input_ids, _, _ in examples)
        pad_id = self.word_idx[self.config.pad_token]

        input_batch: list[list[int]] = []
        segment_batch: list[list[int]] = []
        attention_batch: list[list[int]] = []
        label_batch: list[list[int]] = []
        for input_ids, segment_ids, labels in examples:
            padding = max_length - len(input_ids)
            input_batch.append(input_ids + [pad_id] * padding)
            segment_batch.append(segment_ids + [0] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
            label_batch.append(labels + [-100] * padding)

        device = self.config.device
        return (
            torch.tensor(input_batch, dtype=torch.long, device=device),
            torch.tensor(segment_batch, dtype=torch.long, device=device),
            torch.tensor(attention_batch, dtype=torch.long, device=device),
            torch.tensor(label_batch, dtype=torch.long, device=device),
        )

    def _tokens_from_text(self, text: Any) -> list[str]:
        if hasattr(text, "get_words"):
            return list(text.get_words())
        normalized = str(text).lower() if self.config.lowercase else str(text)
        return re.findall(r"\w+|[^\w\s]", normalized)

    def _tokens_with_specials(self, text: Any) -> list[str]:
        tokens = self._tokens_from_text(text)
        room_for_text = self.config.max_position_embeddings
        if self.config.add_special_tokens:
            room_for_text -= 2
        tokens = tokens[:room_for_text]
        if not self.config.add_special_tokens:
            return tokens
        return [self.config.cls_token, *tokens, self.config.sep_token]

    def _token_ids(self, tokens: list[str]) -> list[int]:
        unk_id = self.word_idx[self.config.unk_token]
        return [self.word_idx.get(token, unk_id) for token in tokens]

    def _segment_ids(self, tokens: list[str]) -> list[int]:
        segment = 0
        segment_ids: list[int] = []
        for token in tokens:
            segment_ids.append(segment)
            if token == self.config.sep_token:
                segment = 1
        return segment_ids

    def _pool(self, tokens: list[str], hidden_states: Tensor, pooling: str) -> Vector:
        if pooling == "cls":
            return hidden_states[0].tolist()
        if pooling != "mean":
            raise ValueError("pooling must be 'mean' or 'cls'.")

        special = {self.config.cls_token, self.config.sep_token, self.config.pad_token}
        content_positions = [
            index for index, token in enumerate(tokens) if token not in special
        ]
        if not content_positions:
            content_positions = list(range(hidden_states.shape[0]))
        return hidden_states[content_positions].mean(dim=0).tolist()

    @staticmethod
    def _attention_matrix(attention: Tensor) -> Matrix:
        if not attention.numel():
            return []
        matrix = attention.tolist()
        normalized: Matrix = []
        for row in matrix:
            total = sum(row)
            normalized.append([value / total for value in row] if total else row)
        return normalized

    def _check_is_fit(self) -> None:
        if self.encoder is None or not self.word_idx:
            raise ValueError("Call fit(corpus) before transform or encode.")


def fit_bert_embeddings(
    corpus: list[Any],
    config: BERTConfig | None = None,
    *,
    epochs: int | None = None,
    pooling: str = "mean",
) -> tuple[BERT, list[BERTEncoderOutput]]:
    """Fit a trainable BERT encoder and return the encoded corpus."""

    model = BERT(config)
    return model, model.fit_transform(corpus, epochs=epochs, pooling=pooling)
