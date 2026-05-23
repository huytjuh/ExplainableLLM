import math

from attention import scaled_dot_product_attention, softmax
from inference import generate_tokens, greedy_decode
from tokenizer import SimpleTokenizer
from training import cross_entropy_loss, perplexity
from transformer import TinyDecoderBlock


def test_tokenizer_round_trip():
    tokenizer = SimpleTokenizer().fit(["LLMs generate tokens."])
    ids = tokenizer.encode("LLMs generate tokens.")
    assert tokenizer.decode(ids) == "llms generate tokens."


def test_softmax_sums_to_one():
    values = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(values), 1.0)
    assert values[2] > values[1] > values[0]


def test_causal_attention_masks_future_positions():
    _, weights = scaled_dot_product_attention(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        causal=True,
    )
    assert weights[0][1] == 0.0


def test_training_helpers():
    loss = cross_entropy_loss([0.1, 0.8, 0.1], 1)
    assert loss < 0.3
    assert perplexity(loss) > 1.0


def test_generation_trace():
    id_to_token = {0: "a", 1: "b"}

    def next_logits(_generated_ids):
        return [0.1, 2.0]

    steps = generate_tokens(next_logits, id_to_token, max_tokens=2, temperature=0.0)
    assert [step.selected_token for step in steps] == ["b", "b"]
    assert greedy_decode([0.1, 2.0]) == 1


def test_tiny_decoder_block_returns_attention_weights():
    output = TinyDecoderBlock().forward([[1.0, 0.0], [0.0, 1.0]])
    assert len(output.hidden_states) == 2
    assert len(output.attention_weights) == 2
