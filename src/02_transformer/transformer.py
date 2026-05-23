"""A tiny decoder-block sketch for tracing shapes and data flow.

Transformer class learning roadmap
----------------------------------

This module keeps the detailed learning sections that the README summarizes
under "Transformer Class: Foundations, Training, and Generation". The goal is
to keep the README easy to scan while preserving implementation-level notes
next to the code they explain.

1. Foundations

- Tokenization: characters, bytes, subwords, BPE, WordPiece, SentencePiece,
  and token IDs.
- Vocabulary design and tradeoffs.
- Embeddings and positional encodings.
- Tensors, shapes, batching, masking, and sequence length.
- The next-token prediction objective.

Deliverables:

- Tokenizer walkthrough notebook.
- Small tokenizer implementation.
- Token-to-logit visual explanation.
- Shape-by-shape transformer diagrams.

2. Transformer Architecture

- Self-attention from query, key, and value projections.
- Multi-head attention.
- Causal masking.
- Feed-forward networks.
- Layer normalization.
- Residual connections.
- Decoder-only transformer blocks.
- Logits, softmax, and token probabilities.

Deliverables:

- Minimal transformer block implementation.
- Attention visualization.
- Mathematical notes for attention and loss.
- Step-by-step forward pass for one prompt.

3. Training Objectives and Optimization

- Language modeling loss.
- Cross-entropy.
- Perplexity.
- Gradient descent, Adam, AdamW, learning-rate schedules, warmup, and weight
  decay.
- Pretraining versus supervised fine-tuning.
- Instruction tuning and preference optimization.
- Overfitting, evaluation splits, and data quality.

Deliverables:

- Toy training loop.
- Loss curve examples.
- Optimizer comparison notes.
- Dataset preparation examples.

4. Inference and Token Generation

- Prefill and decode phases.
- KV cache.
- Temperature.
- Top-k and top-p sampling.
- Greedy decoding and beam search.
- Stop sequences.
- Structured output.
- Streaming responses.
- How each final token is generated from logits and sampling decisions.

Deliverables:

- Token-by-token generation trace.
- Sampling strategy demos.
- KV cache explanation.
- End-to-end prompt-to-output walkthrough.
"""

from __future__ import annotations

from dataclasses import dataclass

from attention import Matrix, scaled_dot_product_attention


@dataclass
class DecoderBlockOutput:
    hidden_states: Matrix
    attention_weights: Matrix


class TinyDecoderBlock:
    """A transparent block: causal attention plus residual connection.

    The projections are identity projections to keep the example focused on
    data flow. Production transformer blocks learn Q/K/V projections, output
    projections, feed-forward layers, and normalization parameters.

    Learning map:
    - Foundations live in `tokenizer.py`, `attention.py`, and shape examples.
    - Architecture is demonstrated by causal attention plus a residual path.
    - Training objectives live in `training.py`.
    - Inference and token generation live in `inference.py`.
    """

    def forward(self, hidden_states: Matrix) -> DecoderBlockOutput:
        attended, weights = scaled_dot_product_attention(
            hidden_states,
            hidden_states,
            hidden_states,
            causal=True,
        )
        residual = [
            [original + update for original, update in zip(row, attended_row)]
            for row, attended_row in zip(hidden_states, attended)
        ]
        return DecoderBlockOutput(hidden_states=residual, attention_weights=weights)
