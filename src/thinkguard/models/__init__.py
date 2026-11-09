"""Load a model for one condition of configs/experiment_registry.yaml: its base, plus its LoRA adapter if it has one."""

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from thinkguard.registry import hub


def load(path, revision=None, adapter=None, device="cuda", dtype=torch.bfloat16):
    """The causal LM at path (a Hugging Face repo id or a local directory) and revision, with the PEFT adapter in the
    directory adapter on top if given, in eval mode, on device, in dtype.
    """
    model = AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=dtype, device_map=device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def load_condition(entry, device="cuda"):
    """load for a registry entry: its base at the revision pinned in the source manifest, and its adapter, a path
    relative to CKPT_DIR unless absolute. Raises KeyError if the entry has an adapter and CKPT_DIR is unset.
    """
    adapter = None if entry.adapter is None else Path(os.environ["CKPT_DIR"]) / entry.adapter
    return load(*hub(entry.base), adapter=adapter, device=device)
