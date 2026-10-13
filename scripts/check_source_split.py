"""Find which BeaverTails split the 15k ThinkGuard training rows came from.

Run from the repo root after scripts/get_data.sh: python scripts/check_source_split.py
Matches conversations on (prompt, response) with whitespace collapsed.
"""
import json
import re

from datasets import get_dataset_split_names, load_dataset

PATH = "data/beavertails_critiques_aug.json"
REPO = "PKU-Alignment/BeaverTails"
INPUT = re.compile(r"^User: (.*?)\nAgent: (.*)$", re.S)


def key(prompt, response):
    return " ".join(prompt.split()), " ".join(response.split())


rows = json.load(open(PATH, encoding="utf-8"))
ours = {key(*INPUT.match(r["input"]).groups()) for r in rows}
print(f"15k file: {len(rows)} rows, {len(ours)} unique (prompt, response) keys")

for split in get_dataset_split_names(REPO):
    ds = load_dataset(REPO, split=split)
    theirs = {key(p, r) for p, r in zip(ds["prompt"], ds["response"])}
    hit = len(ours & theirs)
    print(f"{split:>11}: {len(ds):>7} rows | contains {hit:>6} of {len(ours)} keys ({100 * hit / len(ours):.1f}%)")
