"""Smoke-test vLLM: classify training row 0 and compare with its gold label.

Run from the repo root inside .venv: python scripts/smoke_vllm.py
Exits 1 if the output contains no safe/unsafe label.
"""
import argparse
import json
import re
import sys

from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--data", default="data/beavertails_critiques_aug.json")
    p.add_argument("--dtype", default="auto", help="T4 GPUs need half")
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="the native 262,144 context reserves more KV cache than a 24-40 GB GPU has")
    args = p.parse_args()

    row = json.load(open(args.data, encoding="utf-8"))[0]
    # Same join LLaMA-Factory uses for Alpaca rows
    messages = [{"role": "user", "content": row["instruction"] + "\n" + row["input"]}]
    gold = re.search(r"^- Safety Assessment: (safe|unsafe)\b", row["output"], re.M).group(1)

    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len)
    out = llm.chat(messages, SamplingParams(temperature=0, max_tokens=128))[0]
    text = out.outputs[0].text
    m = re.search(r"\b(unsafe|safe)\b", text, re.I)
    pred = m.group(1).lower() if m else None

    print(f"prompt tokens: {len(out.prompt_token_ids)}")
    print(f"raw output:\n{text}")
    print(f"gold: {gold}")
    print(f"pred: {pred}")
    sys.exit(0 if pred else 1)


# vLLM can spawn worker processes that re-import this file; the guard keeps them from re-running it
if __name__ == "__main__":
    main()
