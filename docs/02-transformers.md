# Transformers

A decoder-only transformer repeatedly transforms token vectors with:

- self-attention to move information across positions,
- causal masks to prevent looking ahead,
- residual connections to preserve signal,
- feed-forward layers to transform each position,
- normalization to stabilize optimization.

The educational implementation in `src/02_transformer/transformer.py` uses identity projections so the shape and attention flow are easy to inspect.
