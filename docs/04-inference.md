# Inference

Inference turns logits into generated tokens.

Main concepts:

- Prefill: process the prompt.
- Decode: generate one token at a time.
- KV cache: reuse previous attention keys and values.
- Greedy decoding: choose the highest-probability token.
- Temperature: flatten or sharpen probabilities.
- Top-k and top-p: restrict the sampling pool.

The `generate_tokens` helper records each generation step with logits, probabilities, and selected token.

