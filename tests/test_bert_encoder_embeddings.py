from __future__ import annotations

from pathlib import Path
import sys


EMBEDDING_DIR = Path(__file__).resolve().parents[1] / "src" / "_01_NLP" / "_04_context_embeddings"
sys.path.insert(0, str(EMBEDDING_DIR))

from _02_bert import BERT, BERTConfig, fit_bert_embeddings


CORPUS = [
    "bank approved the loan",
    "bank raised the interest rate",
    "river bank flooded after rain",
    "support refunded the payment",
]


def test_bert_encoder_returns_contextual_token_embeddings():
    model = BERT(BERTConfig(embedding_dim=8, intermediate_dim=12, seed=7)).fit(CORPUS)

    finance = model.encode("bank approved loan")
    river = model.encode("river bank flooded")
    finance_bank = finance.token_embeddings[1]
    river_bank = river.token_embeddings[2]

    assert finance.tokens == ["[CLS]", "bank", "approved", "loan", "[SEP]"]
    assert len(finance_bank.context_embedding) == 8
    assert len(finance_bank.self_attention) == len(finance.tokens)
    assert finance_bank.word_embedding == river_bank.word_embedding
    assert finance_bank.context_embedding != river_bank.context_embedding


def test_bert_encoder_attention_is_bidirectional():
    model = BERT(BERTConfig(embedding_dim=8, intermediate_dim=12, seed=11)).fit(CORPUS)

    output = model.encode("support refunded payment")
    first_content_token_attention = output.token_embeddings[1].self_attention

    assert len(output.sequence_embedding) == 8
    assert abs(sum(first_content_token_attention) - 1.0) < 1e-9
    assert first_content_token_attention[-1] > 0.0


def test_fit_bert_embeddings_helper_fits_and_transforms():
    model, outputs = fit_bert_embeddings(
        CORPUS,
        BERTConfig(embedding_dim=6, intermediate_dim=10, seed=5),
        pooling="cls",
    )

    assert model.vocabulary_["bank"] >= 0
    assert len(outputs) == len(CORPUS)
    assert outputs[0].sequence_embedding == outputs[0].token_embeddings[0].context_embedding
