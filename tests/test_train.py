"""thinkguard.train on the tiny model (CPU): only LoRA weights train, and the loss of an optimizer step does not
depend on how its rows are split into microbatches (token_mean_all across gradient accumulation, padding ignored).
"""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from peft import LoraConfig  # noqa: E402

from thinkguard.config import load  # noqa: E402
from thinkguard.models import load as load_model  # noqa: E402
from thinkguard.paths import ROOT  # noqa: E402
from thinkguard.train import LORA, TARGETS, add_lora, make_trainer, trainable_params  # noqa: E402
from thinkguard.training.config import TrainConfig  # noqa: E402
from thinkguard.training.data import IGNORE  # noqa: E402

CONFIG = load(ROOT / "configs/train/critique.yaml", TrainConfig)
CPU = {"use_cpu": True, "bf16": False}


def lora_model(path, dropout=0.0):
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=dropout, target_modules=TARGETS, task_type="CAUSAL_LM")
    return add_lora(load_model(path, device="cpu", dtype=torch.float32), 0, lora)


def synthetic_rows():
    """4 rows of different lengths whose last k tokens are supervised, k far from proportional to the length."""
    out = []
    for n, k in [(10, 2), (40, 30), (12, 3), (30, 20)]:
        ids = [(7 * n + 13 * i) % 1000 + 1000 for i in range(n)]
        out.append({"input_ids": ids, "labels": [IGNORE] * (n - k) + ids[n - k :], "length": n})
    return out


def test_trainable_params_are_lora_only(tiny_model):
    model = lora_model(tiny_model)
    report = trainable_params(model)
    assert report["total"] > 0 and all("lora_" in n for n in report["params"])
    model.get_base_model().lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="not LoRA"):
        trainable_params(model)
    assert LORA.r == 16 and LORA.lora_alpha == 32 and LORA.lora_dropout == 0.05  # protocol.md, Training


def test_l_and_c_start_from_the_same_lora_weights(tiny_model):
    first, second = (dict(lora_model(tiny_model).named_parameters()) for _ in range(2))  # the RNG moved in between
    assert all(torch.equal(first[n], second[n]) for n in first if "lora_A" in n)


@pytest.mark.parametrize("microbatch, grad_accum", [(1, 4), (2, 2)])
def test_one_step_is_the_same_however_rows_are_split_into_microbatches(tiny_model, tmp_path, microbatch, grad_accum):
    """One SGD step (lr 1, so the update is the gradient) over the same 4 rows, as one padded microbatch of 4 or as
    microbatches with gradient accumulation, changes the LoRA weights by the same amount.
    """
    step = {**CPU, "optim": "sgd", "learning_rate": 1.0, "warmup_ratio": 0.0, "lr_scheduler_type": "constant",
            "max_steps": 1, "group_by_length": False}  # fmt: skip

    def train(mb, accum, out):
        model = lora_model(tiny_model)
        config = dataclasses.replace(CONFIG, microbatch=mb, grad_accum=accum)
        make_trainer(model, config, synthetic_rows(), out, **step).train()
        return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}

    whole, split = train(4, 1, tmp_path / "whole"), train(microbatch, grad_accum, tmp_path / "split")
    assert max((whole[n] - split[n]).abs().max().item() for n in whole) < 1e-5
    assert max(w.abs().max().item() for n, w in whole.items() if "lora_B" in n) > 1e-3  # the step changed weights


def test_l_and_c_rows_come_in_the_same_batches_and_order(tiny_model, tmp_path):
    """Rows grouped by the same length column (C's) and shuffled by the same seed: L's rows are shorter, but each batch
    holds the same examples. The first token of a row is its example's number.
    """
    c_rows = [{"input_ids": [i] * n, "labels": [IGNORE] * n, "length": n} for i, n in enumerate(range(5, 77, 3))]
    l_rows = [r | {"input_ids": r["input_ids"][:4], "labels": r["labels"][:4]} for r in c_rows]

    def order(rows):
        trainer = make_trainer(lora_model(tiny_model), dataclasses.replace(CONFIG, microbatch=4), rows, tmp_path, **CPU)
        return [b["input_ids"][:, 0].tolist() for b in trainer.get_train_dataloader()]

    c_order = order(c_rows)
    assert order(l_rows) == c_order and sorted(sum(c_order, [])) == list(range(24))
