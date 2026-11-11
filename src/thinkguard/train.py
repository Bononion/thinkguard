"""Train L or C (protocol.md, Training): B plus a LoRA adapter, with Hugging Face's Trainer.

Run from the repo root: python -m thinkguard.train --config configs/train/critique.yaml [--resume $THINKGUARD_RUNS/<id>]
Writes $THINKGUARD_RUNS/<id>/: manifest.json, trainable_params.json, checkpoint-<step>/ at every scored step (the
optimizer state in the latest only), validation/step-<n>.jsonl (the predictions scored) and best_checkpoint.json
(protocol.md, Checkpoint selection). The loss is
Hugging Face's causal-LM loss on the rows' labels, summed over an optimizer step's tokens and divided by their number
(protocol.md, Loss convention). --resume goes on from the run's last checkpoint (optimizer, scheduler, RNG, data
position) after a disconnect.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed

from thinkguard import registry, runs
from thinkguard.config import load
from thinkguard.data.manifest import load_split
from thinkguard.evaluate import EvalConfig
from thinkguard.evaluation.generate import predict
from thinkguard.evaluation.metrics import safety_metrics
from thinkguard.evaluation.parse import parse
from thinkguard.models import load_condition
from thinkguard.prompts import encode
from thinkguard.training.config import MODES, TrainConfig
from thinkguard.training.data import collate, rows

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")


class GuardTrainer(Trainer):
    """A Trainer whose evaluate() is the checkpoint-selection metric: decision-only generation on eval_examples by
    evaluation.generate.predict, the evaluate command's path, parsed and scored by safety_metrics.
    """

    def __init__(self, *args, eval_examples=(), eval_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples, self.eval_config = list(eval_examples), eval_config
        self.eval_prompts = [encode(e, "infer").input_ids for e in self.eval_examples]

    def evaluate(self, *args, **kwargs):
        """{eval_unsafe_f1, eval_coverage} of the model now, logged; predictions in validation/step-<n>.jsonl."""
        path = Path(self.args.output_dir) / "validation" / f"step-{self.state.global_step}.jsonl"
        path.parent.mkdir(exist_ok=True)
        start = time.perf_counter()
        self.model.eval()
        predict(self.model, [e.example_id for e in self.eval_examples], self.eval_prompts, self.eval_config, path)
        self.model.train()
        lines = path.read_text(encoding="utf-8").splitlines()
        by_id = {p["example_id"]: p for p in map(json.loads, lines)}
        scored = ((e.gold_safety_label, by_id[e.example_id]) for e in self.eval_examples)
        m = safety_metrics((gold, parse(p["text"]).label, p["unsafe_score"]) for gold, p in scored)
        metrics = {"eval_unsafe_f1": m["unsafe_f1"], "eval_coverage": m["coverage"]}
        metrics["eval_runtime"] = round(time.perf_counter() - start, 1)  # seconds, for the pilot (#61)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class Checkpoints(TrainerCallback):
    """Scores and saves at the last step too, and deletes the optimizer state of every checkpoint but the latest,
    which --resume needs (about 0.3 GB each for B).
    """

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        saved = sorted(Path(args.output_dir).glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[1]))
        for d in saved[:-1]:
            (d / "optimizer.pt").unlink(missing_ok=True)


def trainable_params(model):
    """{params: {name: shape}, total} of model's trainable parameters. Raises ValueError if one is not a LoRA weight."""
    params = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    other = [n for n in params if "lora_" not in n]
    if other:
        raise ValueError(f"{len(other)} trainable parameters are not LoRA weights, first {other[:3]}")
    return {"params": params, "total": sum(math.prod(s) for s in params.values())}


def add_lora(model, seed, lora=LORA):
    """model with a new LoRA adapter, initialized after set_seed(seed) so L and C start from the same weights."""
    set_seed(seed)
    return get_peft_model(model, lora)


def make_trainer(model, config, train_rows, output_dir, eval_examples=(), **overrides):
    """A GuardTrainer for model (B with LoRA) on train_rows (training.data.rows) with protocol.md's settings; overrides
    replace TrainingArguments (the tests run on CPU without bf16). With eval_examples, it scores and saves every
    ceil(steps per epoch / evals_per_epoch) optimizer steps and at the last, and tracks the best eval_unsafe_f1 (a tie
    keeps the earlier step).
    """
    args = dict(
        output_dir=str(output_dir), per_device_train_batch_size=config.microbatch,
        gradient_accumulation_steps=config.grad_accum, learning_rate=config.lr, num_train_epochs=config.epochs,
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="adamw_torch", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, group_by_length=True, seed=config.seed,
        data_seed=config.seed, logging_steps=1, include_num_input_tokens_seen=True, remove_unused_columns=False,
        report_to=[], save_strategy="no",
    )  # fmt: skip
    eval_config, callbacks = None, []
    if eval_examples:
        steps = math.ceil(math.ceil(len(train_rows) / config.microbatch) / config.grad_accum)
        every = math.ceil(steps / config.evals_per_epoch)
        args |= dict(eval_strategy="steps", eval_steps=every, save_strategy="steps", save_steps=every,
                     metric_for_best_model="unsafe_f1", greater_is_better=True)  # fmt: skip
        eval_config = EvalConfig(MODES[config.mode], config.eval_split, "decision", max_new_tokens=128, batch_size=16)
        callbacks = [Checkpoints()]
    return GuardTrainer(
        model=model, args=TrainingArguments(**args | overrides), train_dataset=Dataset.from_list(train_rows),
        data_collator=collate, callbacks=callbacks, eval_examples=eval_examples, eval_config=eval_config,
        eval_dataset=list(eval_examples) or None,  # HF wants one with eval_strategy; evaluate() reads eval_examples
    )  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train L or C.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", type=Path, help="the run directory of an interrupted run")
    args = parser.parse_args(argv)
    config = load(args.config, TrainConfig)
    train_rows = rows(load_split(config.train_split), config.mode, config.max_length)
    splits = (config.train_split, config.eval_split)
    with runs.start_run("train", config, splits=splits, parent="B", run_dir=args.resume) as run_dir:
        print(run_dir, flush=True)  # the id for --resume
        model = add_lora(load_condition(registry.load()["B"]), config.seed)
        report = trainable_params(model)
        (run_dir / "trainable_params.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        trainer = make_trainer(model, config, train_rows, run_dir, load_split(config.eval_split))
        trainer.train(resume_from_checkpoint=get_last_checkpoint(run_dir))
        best = {"checkpoint": trainer.state.best_model_checkpoint, "eval_unsafe_f1": trainer.state.best_metric}
        (run_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=1) + "\n", encoding="utf-8")
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)  # for the pilot


if __name__ == "__main__":
    main()
