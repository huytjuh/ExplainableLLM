# Training

Language models are commonly trained with next-token prediction.

For each position, the model predicts a probability distribution over the vocabulary. Cross-entropy penalizes the model when the correct next token receives low probability:

```text
loss = -log(p(correct_token))
```

Perplexity is `exp(loss)` and can be read as the model's average uncertainty over next-token choices.

