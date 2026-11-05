"""Spike #35: one LoRA step on B, then save the adapter, reload it on a fresh base and compare logits.

Run from the repo root on the GPU host: uv run python scripts/spike_lora.py [--model llama-guard-3-1b]
Prints the "LoRA" section for reports/compat_spike.md and writes it to runs/<id>/lora.md, with the adapter in
runs/<id>/adapter/. Raises if a trainable parameter is not LoRA, the LoRA parameter count differs from the config's,
lora_B did not move, a base weight changed, or the reloaded adapter changes an argmax or moves logits as much as the
adapter itself does. The step trains on critique row 0 (prompt tokens masked); logits are compared on rows 0-2.
"""

import argparse
import gc
import json
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from spike_load import TARGETS, peak
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkguard.data.sources import load
from thinkguard.paths import ROOT
from thinkguard.runs import environment, start_run


@dataclass(frozen=True)
class Spike:
    model: str
    revision: str
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 1e-3  # one step must move the logits visibly; not the training lr
    rows: int = 3


def expected_lora_params(c, r):
    """r * sum(in + out) over the 7 TARGETS of every layer, from the model config."""
    head = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    h, q, kv = c.hidden_size, c.num_attention_heads * head, c.num_key_value_heads * head
    return r * c.num_hidden_layers * (2 * (h + q) + 2 * (h + kv) + 3 * (h + c.intermediate_size))


def base(config):
    return AutoModelForCausalLM.from_pretrained(
        config.model, revision=config.revision, dtype=torch.bfloat16, device_map="cuda"
    )


def logits(model, prompts):
    """Float32 CPU logits of each prompt, one forward per prompt (batch 1, no padding)."""
    with torch.no_grad():
        return [model(input_ids=p[None]).logits[0].float().cpu() for p in prompts]


def max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b, strict=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-guard-3-8b", choices=("llama-guard-3-1b", "llama-guard-3-8b"))
    name = parser.parse_args().model
    with open(ROOT / "configs/frozen/source_manifest.json", encoding="utf-8") as f:
        source = next(s for s in json.load(f) if s["name"] == name)
    config = Spike(source["url"].removeprefix("https://huggingface.co/"), source["revision"])
    rows = load("critique")[: config.rows]
    with start_run("spike", config) as run_dir:
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        torch.cuda.reset_peak_memory_stats()
        lora = LoraConfig(
            r=config.r, lora_alpha=config.alpha, lora_dropout=config.dropout, target_modules=list(TARGETS)
        )
        model = get_peft_model(base(config), lora)
        load_peak = peak()

        def ids(text, special=True):
            return tokenizer(text, add_special_tokens=special, return_tensors="pt").input_ids[0].to(model.device)

        prompts = [ids(r["instruction"] + "\n\n" + r["input"]) for r in rows]
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if bad := [n for n in trainable if "lora_" not in n]:
            raise RuntimeError(f"trainable non-LoRA parameters: {bad[:5]}")
        count, expected = sum(p.numel() for p in trainable.values()), expected_lora_params(model.config, config.r)
        if count != expected:
            raise RuntimeError(f"{count} LoRA parameters, config gives {expected}")
        bases = [n for n, p in model.named_parameters() if "base_layer" in n]
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if n in (bases[0], bases[-1])}

        x = torch.cat([prompts[0], ids(rows[0]["output"], special=False)])[None]
        labels = x.clone()
        labels[0, : len(prompts[0])] = -100
        tokens = x.shape[1]
        optimizer = torch.optim.AdamW(trainable.values(), lr=config.lr)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        loss = model(input_ids=x, labels=labels).loss
        loss.backward()
        a_grad_zero = all(not p.grad.any() for n, p in trainable.items() if "lora_A" in n)
        optimizer.step()
        train_peak = peak()
        if not all(p.any() for n, p in trainable.items() if "lora_B" in n):
            raise RuntimeError("a lora_B tensor is still zero after the step")
        params = dict(model.named_parameters())
        if not all(torch.equal(t, params[n]) for n, t in frozen.items()):
            raise RuntimeError("a base weight changed")
        dtypes = sorted({str(p.dtype) for p in trainable.values()})

        model.eval()
        reference = logits(model, prompts)
        with model.disable_adapter():
            effect = max_diff(reference, logits(model, prompts))
        model.save_pretrained(run_dir / "adapter")
        loss = loss.item()
        del model, optimizer, trainable, frozen, params, x, labels
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        model = PeftModel.from_pretrained(base(config), run_dir / "adapter").eval()
        reload_peak = peak()
        reloaded = logits(model, prompts)
        reload = max_diff(reference, reloaded)
        same = all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(reference, reloaded, strict=True))
        if not same or reload >= effect:
            raise RuntimeError(f"reloaded adapter: argmax equal {same}, max |diff| {reload} vs adapter effect {effect}")

        env = environment() | {"cuda": torch.version.cuda}
        lines = [
            f"## LoRA ({name})",
            "",
            f"Run `{run_dir.name}`, `{config.model}` at `{config.revision}`, bf16 base.",
            "",
            "| Field | Value |",
            "|---|---|",
            *(f"| {k} | {v} |" for k, v in env.items()),
            f"| r / alpha / dropout | {config.r} / {config.alpha} / {config.dropout} |",
            f"| targets | {', '.join(TARGETS)} |",
            f"| trainable parameters | {count:,} (config gives {expected:,}) |",
            f"| adapter dtype | {', '.join(dtypes)} |",
            f"| step tokens, prompt + output | {len(prompts[0])} + {tokens - len(prompts[0])} |",
            f"| loss | {loss:.4f} |",
            f"| lora_A grad all zero | {a_grad_zero} |",
            f"| adapter effect, max abs logit diff | {effect:.4g} |",
            f"| reload vs trained, max abs logit diff | {reload:.4g} |",
            "",
            f"Peak memory, allocated / reserved GiB: load {load_peak}, train step {train_peak}, reload {reload_peak}.",
        ]
        text = "\n".join(lines) + "\n"
        (run_dir / "lora.md").write_text(text, encoding="utf-8", newline="\n")
        print(text)


if __name__ == "__main__":
    main()
