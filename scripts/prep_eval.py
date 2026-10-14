"""Prepare evaluation test sets under data/test/.

Run from the repo root after scripts/get_data.sh: python scripts/prep_eval.py beavertails
"""
import argparse
import json
import os
import re

from datasets import load_dataset

TRAIN = "data/beavertails_critiques_aug.json"
INPUT = re.compile(r"^User: (.*?)\nAgent: (.*)$", re.S)


# Same conversation key as scripts/check_source_split.py
def key(prompt, response):
    return " ".join(prompt.split()), " ".join(response.split())


def beavertails():
    out_path = "data/test/beavertails.jsonl"
    train = {key(*INPUT.match(r["input"]).groups()) for r in json.load(open(TRAIN, encoding="utf-8"))}
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    rows = [{"id": f"beavertails-{i}", "prompt": r["prompt"], "response": r["response"],
             "label": {True: "safe", False: "unsafe"}.get(r["is_safe"])}
            for i, r in enumerate(ds) if key(r["prompt"], r["response"]) not in train]

    dropped = len(ds) - len(rows)
    assert dropped == 7, f"dropped {dropped} rows, expected 7"
    assert len(rows) == 3014, f"{len(rows)} rows, expected 3014"
    assert all(r["prompt"].strip() and r["response"].strip() for r in rows), "empty prompt or response"
    assert all(r["label"] in ("safe", "unsafe") for r in rows), "is_safe is not a bool"
    assert not any(key(r["prompt"], r["response"]) in train for r in rows), "a written key is in the training file"

    os.makedirs("data/test", exist_ok=True)
    # newline="\n": the same bytes on Windows and Linux
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    unsafe = sum(r["label"] == "unsafe" for r in rows)
    print(f"wrote {out_path}: {len(rows)} rows ({unsafe} unsafe), dropped {dropped} that overlap training")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["beavertails"])
    args = p.parse_args()
    if args.dataset == "beavertails":
        beavertails()
