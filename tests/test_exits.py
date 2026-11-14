"""thinkguard.models.exits on the tiny model (conftest.py, 2 layers), on CPU in float32."""

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig, get_peft_model  # noqa: E402

from thinkguard.models import load  # noqa: E402
from thinkguard.models.exits import exit_logits  # noqa: E402

IDS = torch.tensor([[128000, 20451, 6220, 1000, 128009]])


def test_the_last_exit_is_the_models_logits_and_positions_select(model):
    with torch.no_grad():
        out = model(IDS, output_hidden_states=True)
        mask = torch.tensor([[False, True, False, True, True]])
        assert torch.equal(exit_logits(model, out.hidden_states, 2), out.logits)
        selected = exit_logits(model, out.hidden_states, 1, mask)
        assert torch.allclose(selected, exit_logits(model, out.hidden_states, 1)[mask], atol=1e-6)
    with pytest.raises(ValueError, match="exit 3"):
        exit_logits(model, out.hidden_states, 3)


def test_an_exit_matches_the_model_cut_at_that_depth(model):
    """generate(assistant_early_exit=h) drafts with config.num_hidden_layers = h: the same logits as exit h."""
    with torch.no_grad():
        exit_1 = exit_logits(model, model(IDS, output_hidden_states=True).hidden_states, 1)
        model.config.num_hidden_layers = 1
        try:
            cut = model(IDS).logits
        finally:
            model.config.num_hidden_layers = 2
    assert torch.allclose(exit_1, cut, atol=1e-6)


def test_an_exit_loss_trains_only_the_lora_below_it(tiny_model):
    lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
    model = get_peft_model(load(tiny_model, device="cpu", dtype=torch.float32), lora)
    out = model(IDS, output_hidden_states=True)
    logits = exit_logits(model, out.hidden_states, 1)[0, :-1]
    torch.nn.functional.cross_entropy(logits, IDS[0, 1:]).backward()
    grads = {n: p.grad for n, p in model.named_parameters() if "lora_" in n}
    assert all(g is not None and g.abs().sum() > 0 for n, g in grads.items() if ".layers.0." in n)
    assert all(g is None or not g.any() for n, g in grads.items() if ".layers.1." in n)
