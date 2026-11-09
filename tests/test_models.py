"""thinkguard.models.load on the tiny model (conftest.py), on CPU in float32."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from thinkguard.models import load

IDS = torch.tensor([[128000, 20451, 6220, 128009]])


def logits(model):
    with torch.no_grad():
        return model(IDS).logits


def test_load_gives_the_base_and_attaches_an_adapter(tiny_model, tmp_path):
    base = AutoModelForCausalLM.from_pretrained(tiny_model, dtype=torch.float32)
    torch.manual_seed(1)  # init_lora_weights=False: B is random, not 0, so the adapter changes the logits
    lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    with_lora = get_peft_model(base, lora)
    expected = logits(with_lora)
    with_lora.save_pretrained(tmp_path / "adapter")

    plain = load(tiny_model, device="cpu", dtype=torch.float32)
    assert not plain.training
    with torch.no_grad(), with_lora.disable_adapter():
        assert torch.equal(logits(plain), logits(with_lora))
    adapted = load(tiny_model, adapter=tmp_path / "adapter", device="cpu", dtype=torch.float32)
    assert torch.equal(logits(adapted), expected)
    assert not torch.allclose(logits(adapted), logits(plain))
