# Tokenization

Tokenization converts text into model-readable ids.

Key steps:

- Normalize text.
- Split into pieces: words, bytes, characters, or subwords.
- Map each piece to a vocabulary id.
- Add control tokens such as BOS and EOS.

See `src/02_transformer/tokenizer.py` for a tiny implementation that makes this process visible.
