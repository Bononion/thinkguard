"""The prompt contract (reports/prompt_contract.md, Contract): one serialization for training and evaluation."""

import functools

from transformers import AutoTokenizer

from thinkguard.registry import hub

BASE = "llama-guard-3-8b"  # B; L and C are B plus a LoRA, so they share its tokenizer


@functools.cache
def tokenizer():
    """B's tokenizer at the revision pinned in configs/frozen/source_manifest.json. Needs a Hugging Face login with
    access to the gated repo; loaded once per process.
    """
    repo, revision = hub(BASE)
    return AutoTokenizer.from_pretrained(repo, revision=revision)
