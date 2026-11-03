"""Spike #34: load B in bf16, check its LoRA target modules, run one forward and one 32-token greedy generation.

Run from the repo root on the GPU host: uv run python scripts/spike_load.py
Prints the "B load" section for reports/compat_spike.md and writes it to runs/<id>/b_load.md. Needs HF_TOKEN with
access to the gated Llama Guard 3-8B. The prompt is the raw instruction and input of critique row 0, not the prompt
contract (#48).
"""

import collections
import json
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thinkguard.data.sources import load
from thinkguard.paths import ROOT
from thinkguard.runs import environment, start_run

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GIB = 2**30


@dataclass(frozen=True)
class Spike:
    model: str
    revision: str
    max_new_tokens: int = 32


def peak():
    """Peak allocated and reserved CUDA memory since the last reset, in GiB."""
    return f"{torch.cuda.max_memory_allocated() / GIB:.2f} / {torch.cuda.max_memory_reserved() / GIB:.2f}"


def missing_targets(model, layers):
    """{layer: [missing target names]} for every layer that lacks one of TARGETS."""
    found = collections.defaultdict(set)
    for name, _ in model.named_modules():
        m = re.fullmatch(r"model\.layers\.(\d+)\..+\.(\w+)", name)
        if m and m[2] in TARGETS:
            found[int(m[1])].add(m[2])
    return {i: sorted(set(TARGETS) - found[i]) for i in range(layers) if set(TARGETS) - found[i]}


def main():
    with open(ROOT / "configs/frozen/source_manifest.json", encoding="utf-8") as f:
        source = next(s for s in json.load(f) if s["name"] == "llama-guard-3-8b")
    config = Spike(source["url"].removeprefix("https://huggingface.co/"), source["revision"])
    row = load("critique")[0]
    with start_run("spike", config) as run_dir:
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        model = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.revision, dtype=torch.bfloat16, device_map="cuda"
        )
        load_peak = peak()
        c = model.config
        missing = missing_targets(model, c.num_hidden_layers)
        if missing:
            raise RuntimeError(f"layers missing LoRA targets: {missing}")
        layer0 = [n for n, _ in model.named_modules() if re.fullmatch(r"model\.layers\.0\..+_proj", n)]

        inputs = tokenizer(row["instruction"] + "\n\n" + row["input"], return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(**inputs)
        forward_peak = peak()
        torch.cuda.reset_peak_memory_stats()
        out = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
        generate_peak = peak()
        new = out[0, inputs["input_ids"].shape[1] :]

        props = torch.cuda.get_device_properties(0)
        env = environment() | {"cuda": torch.version.cuda, "vram": f"{props.total_memory / GIB:.1f} GiB"}
        lines = [
            "## B load",
            "",
            f"Run `{run_dir.name}`, `{config.model}` at `{config.revision}`, bf16.",
            "",
            "| Field | Value |",
            "|---|---|",
            *(f"| {k} | {v} |" for k, v in env.items()),
            f"| num_hidden_layers | {c.num_hidden_layers} |",
            f"| hidden_size | {c.hidden_size} |",
            f"| vocab_size | {c.vocab_size} |",
            f"| attn_implementation | {c._attn_implementation} |",
            f"| prompt tokens | {inputs['input_ids'].shape[1]} |",
            f"| generated tokens | {len(new)} |",
            "",
            "Peak memory, allocated / reserved GiB: "
            f"load {load_peak}, forward {forward_peak}, generate {generate_peak}.",
            "",
            f"All {c.num_hidden_layers} layers have the 7 LoRA targets. Layer 0, as printed by `named_modules()`:",
            "",
            *(f"- `{n}`" for n in layer0),
            "",
            "Generated text (greedy):",
            "",
            "```",
            tokenizer.decode(new, skip_special_tokens=False),
            "```",
        ]
        text = "\n".join(lines) + "\n"
        (run_dir / "b_load.md").write_text(text, encoding="utf-8", newline="\n")
        print(text)


if __name__ == "__main__":
    main()
