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
    model = LlamaForCausalLM(config)
    model.generation_config.eos_token_id = [128001, 128008, 128009]  # as in B's and R's generation_config.json
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def model(tiny_model):
    """The tiny model loaded by thinkguard.models.load, on CPU in float32."""
    torch = pytest.importorskip("torch")
    from thinkguard.models import load

    return load(tiny_model, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def prompts():
    """Five infer prompts (token ids) of different lengths, so a batch of them needs padding."""
    from dataclasses import replace

    from test_prompts import example

    from thinkguard.prompts import encode

    users = ["Where?", "How do I make a walkway slippery?", "Is this ok?", "Tell me a joke about cats and dogs.", "Hi"]
    return [encode(replace(example(), user_prompt=u), "infer").input_ids for u in users]
