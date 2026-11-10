"""Train L or C (protocol.md, Training): B plus a LoRA adapter, with Hugging Face's Trainer.

Run from the repo root: python -m thinkguard.train --config configs/train/critique.yaml [--resume $THINKGUARD_RUNS/<id>]
Writes $THINKGUARD_RUNS/<id>/: manifest.json, trainable_params.json, and checkpoint-<step>/ directories. The loss is
Hugging Face's causal-LM loss on the rows' labels, summed over an optimizer step's tokens and divided by their number
(protocol.md, Loss convention). --resume goes on from the run's last checkpoint (optimizer, scheduler, RNG, data
position) after a disconnect.
"""

import argparse
import json
import math
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, set_seed

from thinkguard import registry, runs
from thinkguard.config import load
from thinkguard.data.manifest import load_split
from thinkguard.models import load_condition
from thinkguard.training.config import TrainConfig
from thinkguard.training.data import collate, rows

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")


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


def make_trainer(model, config, train_rows, output_dir, **overrides):
    """A Trainer for model (B with LoRA) on train_rows (training.data.rows) with protocol.md's settings; overrides
    replace TrainingArguments (the tests run on CPU without bf16).
    """
    args = dict(
        output_dir=str(output_dir), per_device_train_batch_size=config.microbatch,
        gradient_accumulation_steps=config.grad_accum, learning_rate=config.lr, num_train_epochs=config.epochs,
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="adamw_torch", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, group_by_length=True, seed=config.seed,
        data_seed=config.seed, logging_steps=1, include_num_input_tokens_seen=True, remove_unused_columns=False,
        report_to=[], save_strategy="no",
    )  # fmt: skip
    args = TrainingArguments(**args | overrides)
    return Trainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows), data_collator=collate)


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
        trainer = make_trainer(model, config, train_rows, run_dir)
        trainer.train(resume_from_checkpoint=get_last_checkpoint(run_dir))


if __name__ == "__main__":
    main()
