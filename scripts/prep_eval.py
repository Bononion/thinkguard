"""Prepare evaluation sets under data/test/ and data/val/.

Run from the repo root after scripts/get_data.sh:
    python scripts/prep_eval.py beavertails       # test set
    python scripts/prep_eval.py beavertails_val   # val set; needs the test set first
"""
import argparse
import json
import os
import random
import re

from datasets import load_dataset

TRAIN = "data/beavertails_critiques_aug.json"
TEST = "data/test/beavertails.jsonl"
VAL = "data/val/beavertails_val.jsonl"
BEAVERTAILS = "PKU-Alignment/BeaverTails"
INPUT = re.compile(r"^User: (.*?)\nAgent: (.*)$", re.S)
SEED = 42
VAL_SIZE = 3000


# Same conversation key as scripts/check_source_split.py
def key(prompt, response):
    return " ".join(prompt.split()), " ".join(response.split())


def train_keys():
    return {key(*INPUT.match(r["input"]).groups()) for r in json.load(open(TRAIN, encoding="utf-8"))}


def to_row(row_id, r):
    return {"id": row_id, "prompt": r["prompt"], "response": r["response"],
            "label": {True: "safe", False: "unsafe"}.get(r["is_safe"])}


def write(path, rows):
    """Check and write rows as JSONL; return the unsafe count."""
    assert all(r["prompt"].strip() and r["response"].strip() for r in rows), "empty prompt or response"
    assert all(r["label"] in ("safe", "unsafe") for r in rows), "is_safe is not a bool"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # newline="\n": the same bytes on Windows and Linux
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return sum(r["label"] == "unsafe" for r in rows)


def beavertails():
    train = train_keys()
    ds = load_dataset(BEAVERTAILS, split="30k_test")
    rows = [to_row(f"beavertails-{i}", r) for i, r in enumerate(ds) if key(r["prompt"], r["response"]) not in train]

    dropped = len(ds) - len(rows)
    assert dropped == 7, f"dropped {dropped} rows, expected 7"
    assert len(rows) == 3014, f"{len(rows)} rows, expected 3014"
    assert not any(key(r["prompt"], r["response"]) in train for r in rows), "a written key is in the training file"

    unsafe = write(TEST, rows)
    print(f"wrote {TEST}: {len(rows)} rows ({unsafe} unsafe), dropped {dropped} that overlap training")


def beavertails_val():
    excluded = train_keys() | {key(r["prompt"], r["response"]) for r in map(json.loads, open(TEST, encoding="utf-8"))}
    ds = load_dataset(BEAVERTAILS, split="30k_train")
    groups = {}
    for i, r in enumerate(ds):
        groups.setdefault(key(r["prompt"], r["response"]), []).append((i, r))
    dups = sum(len(g) > 1 for g in groups.values())
    conflicts = {k for k, g in groups.items() if len({r["is_safe"] for _, r in g}) > 1}
    agreed = [k for k in groups if k not in conflicts]
    pool = sorted(k for k in agreed if k not in excluded)
    # Keep the first row of each key; its index is the id
    rows = [to_row(f"beavertails_val-{groups[k][0][0]}", groups[k][0][1]) for k in random.Random(SEED).sample(pool, VAL_SIZE)]

    keys = {key(r["prompt"], r["response"]) for r in rows}
    assert len(rows) == len(keys) == VAL_SIZE, f"{len(rows)} rows, {len(keys)} unique keys, expected {VAL_SIZE}"
    assert not keys & excluded, "a val key is in the training or test file"

    unsafe = write(VAL, rows)
    print(f"30k_train: {len(ds)} rows, {len(groups)} unique keys; {dups} keys have duplicate rows "
          f"({len(ds) - len(groups)} extra rows), {len(conflicts)} conflicting-label keys dropped")
    print(f"pool: {len(pool)} keys ({len(agreed) - len(pool)} training/test keys removed)")
    print(f"wrote {VAL}: {len(rows)} rows ({unsafe} unsafe), seed {SEED}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["beavertails", "beavertails_val"])
    args = p.parse_args()
    if args.dataset == "beavertails":
        beavertails()
    elif args.dataset == "beavertails_val":
        beavertails_val()
