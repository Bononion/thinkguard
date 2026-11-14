"""Load a model for one condition of configs/experiment_registry.yaml: its base, plus its LoRA adapter if it has one."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from thinkguard.registry import hub


def load(path, revision=None, adapter=None, adapter_revision=None, device="cuda", dtype=torch.bfloat16):
    """The causal LM at path (a Hugging Face repo id or a local directory) and revision, with the PEFT adapter at
    adapter (the same kinds) and adapter_revision on top if given, in eval mode, on device, in dtype.
    """
    model = AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=dtype, device_map=device)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, revision=adapter_revision)
    return model.eval()


def load_condition(entry, device="cuda"):
    """load for a registry entry: its base at the revision pinned in the source manifest, and its adapter: a local
    checkpoint directory (evaluate --adapter), or else a model in the source manifest at its pinned revision.
    """
    adapter = revision = None
    if entry.adapter is not None:
        adapter, revision = (entry.adapter, None) if Path(entry.adapter).is_dir() else hub(entry.adapter)
    return load(*hub(entry.base), adapter=adapter, adapter_revision=revision, device=device)
