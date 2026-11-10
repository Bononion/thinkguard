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
from thinkguard.prompts import encode  # noqa: E402
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


def test_scored_checkpoints_match_a_standalone_predict(tiny_model, tmp_path):
    """5 steps, scored every ceil(5 / 2) = 3 and at the last: checkpoints 3 and 5, the optimizer state in 5 only, and
    checkpoint 3's adapter reloaded and run through predict gives the bytes the training run scored.
    """
    from peft import PeftModel
    from test_prompts import example

    from thinkguard.evaluation.generate import predict

    golds = ["safe", "unsafe", "unsafe", "safe"]
    examples = [dataclasses.replace(example(), example_id=f"v{i}", user_prompt=f"Where {i}?", gold_safety_label=g)
                for i, g in enumerate(golds)]  # fmt: skip
    config = dataclasses.replace(CONFIG, microbatch=1, grad_accum=1, epochs=1, evals_per_epoch=2)
    model = lora_model(tiny_model, dropout=0.5)  # so scoring in train mode would not match
    trainer = make_trainer(model, config, synthetic_rows() + synthetic_rows()[:1], tmp_path, examples, **CPU)
    trainer.train()
    assert sorted(d.name for d in tmp_path.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-5"]
    assert [p.parent.name for p in tmp_path.glob("checkpoint-*/optimizer.pt")] == ["checkpoint-5"]
    assert [h["step"] for h in trainer.state.log_history if "eval_unsafe_f1" in h] == [3, 5]
    assert (trainer.eval_config.mode, trainer.eval_config.max_new_tokens) == ("decision", 128)
    assert trainer.state.best_model_checkpoint in (str(tmp_path / "checkpoint-3"), str(tmp_path / "checkpoint-5"))

    reloaded = PeftModel.from_pretrained(
        load_model(tiny_model, device="cpu", dtype=torch.float32), tmp_path / "checkpoint-3"
    )
    prompts = [encode(e, "infer").input_ids for e in examples]
    predict(
        reloaded.eval(), [e.example_id for e in examples], prompts, trainer.eval_config, tmp_path / "standalone.jsonl"
    )
    assert (tmp_path / "standalone.jsonl").read_bytes() == (tmp_path / "validation/step-3.jsonl").read_bytes()


def test_a_run_interrupted_after_a_checkpoint_resumes_to_the_same_losses_and_weights(tiny_model, tmp_path):
    """6 steps scored and saved at 3 and 6, with LoRA dropout (so the RNG state matters): uninterrupted, and stopped
    by KeyboardInterrupt right after checkpoint 3 is saved, then resumed from it in a new Trainer.
    """
    from test_prompts import example
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint

    class Stop(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 3:
                raise KeyboardInterrupt

    examples = [dataclasses.replace(example(), example_id="v", gold_safety_label="unsafe")]
    config = dataclasses.replace(CONFIG, microbatch=1, grad_accum=1, epochs=1, evals_per_epoch=2)
    train_rows = synthetic_rows() + synthetic_rows()[:2]

    def run(out, callbacks=()):
        trainer = make_trainer(lora_model(tiny_model, dropout=0.5), config, train_rows, out, examples, **CPU)
        for c in callbacks:
            trainer.add_callback(c)
        return trainer

    whole = run(tmp_path / "whole")
    whole.train()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "cut", [Stop()]).train()

    class Count(TrainerCallback):
        steps = 0

        def on_step_end(self, args, state, control, **kwargs):
            Count.steps += 1

    resumed = run(tmp_path / "cut", [Count()])
    resumed.train(resume_from_checkpoint=get_last_checkpoint(tmp_path / "cut"))
    assert Count.steps == 3  # steps 4-6 only: it resumed, not restarted

    def losses(t):
        return [(h["step"], h["loss"]) for h in t.state.log_history if "loss" in h]

    assert len(losses(whole)) == 6 and losses(resumed) == pytest.approx(losses(whole), abs=1e-6)
    w, r = (dict(t.model.named_parameters()) for t in (whole, resumed))
    assert max((w[n] - r[n]).abs().max().item() for n in w if "lora_" in n) < 1e-6
