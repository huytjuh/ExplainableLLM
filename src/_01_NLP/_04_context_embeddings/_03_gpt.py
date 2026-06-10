from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTEmbedding:
    pass

@dataclass 
class GPTConfig:

    dim: int = 2
    max_length: int = 5

class WordEmbedding(nn.Module):
    """Word embedding layer."""
    
    def __init__(self, config: GPTConfig | None = None) -> None:
        """Initialize the word embedding layer."""
        super().__init__()
        self.config = config or GPTConfig()

        self.embeddings = nn.Embedding(self.config.max_length, self.config.dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the word embedding layer."""
        return self.embedding(tokens)

class PositionalEncoding(nn.Module):
    """Add positional encodings to word embeddings."""

    def __init__(self, config: GPTConfig | None = None) -> None:
        """Initialize the positional embedding layer."""
        super().__init__()
        self.config = config or GPTConfig()

        positional_encodings = torch.zeros(self.config.max_length, self.config.dim)

        pos = torch.arange(0, self.config.max_length).unsqueeze(1)
        i = torch.arange(0, self.config.dim // 2)

        div_term = 1/torch.tensor(10000.0)**(i / self.config.dim)

        positional_encodings[:, 0::2] = torch.sin(pos * div_term)
        positional_encodings[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", positional_encodings)

    def forward(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to word embeddings."""
        return word_embeddings + self.positional_encodings

class Attention(nn.Module):
    """Masked self-attention layer."""

    def __init__(self, config: GPTConfig | None = None) -> None:
        """Initialize the masked self-attention layer."""
        super().__init__()
        self.config = config or GPTConfig()

        self.W_Q = nn.Linear(in_features=self.config.dim, out_features=self.config.dim)
        self.W_K = nn.Linear(in_features=self.config.dim, out_features=self.config.dim)
        self.W_V = nn.Linear(in_features=self.config.dim, out_features=self.config.dim)

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_q: torch.Tensor, encodings_k: torch.Tensor, encodings_v: torch.Tensor, mask=None) -> torch.Tensor:
        """Forward pass through the masked self-attention layer."""
        Q = self.W_Q(encodings_q)
        K = self.W_K(encodings_k)
        V = self.W_V(encodings_v)

        sims = torch.matmul(Q, K.transpose(self.row_dim, self.col_dim))
        scaled_sims = sims / torch.tensor(K.size(self.col_dim)**0.5)

        if mask is not None:
            sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_proba = nn.functional.softmax(sims, dim=self.col_dim)
        attention = torch.matmul(attention_proba, V)

        return attention
    

class GPT(nn.Module):
    """GPT model."""

    def __init__(self, config: GPTConfig | None = None) -> None:
        """Initialize the GPT model."""
        super().__init__()
        self.config = config or GPTConfig()

        self.word_embeddings = WordEmbedding(self.config)
        self.positional_encodings = PositionalEncoding(self.config)
        self.self_attention = Attention(self.config)
        self.fc_layer = nn.Linear(in_features=self.config.dim, out_features=self.config.dim)
        
        self.loss = nn.CrossEntropyLoss()


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT model."""
        word_embeddings = self.word_embeddings(tokens)
        positional_encodings = self.positional_encodings(word_embeddings)

        mask = torch.tril(torch.ones(tokens.size(0), tokens.size(0)))
        mask = mask == 0
        self_attention = self.self_attention(positional_encodings, positional_encodings, positional_encodings, mask)

        resid_layer = positional_encodings + self_attention
        fc_layer = self.fc_layer(resid_layer)

        return fc_layer