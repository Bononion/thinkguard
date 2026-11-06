"""Shared fixtures."""

import pytest


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Directory of a 2-layer Llama with random weights (fixed seed) and the real vocab size, so tests can use real
    token ids. Same model class as Llama Guard 3, so the code paths match; the outputs mean nothing.
    """
    torch = pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128256, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=4096, bos_token_id=128000, eos_token_id=128009,
        initializer_range=0.5,  # the default 0.02 gives near-zero logits, so every prompt's unsafe score is about equal
    )  # fmt: skip
    path = tmp_path_factory.mktemp("tiny_model")
    LlamaForCausalLM(config).save_pretrained(path)
    return path
