"""Frozen split manifests: the example_ids of each split, so every run trains and scores on the same rows.

Run from the repo root, after filter_train: python -m thinkguard.data.manifest
Writes data/manifests/<split>.json for train, validation, smoke and pilot (by the rules in protocol.md, Validation
split) and for test_<set>, every row of each test set in sources.TEST_SETS, and prints each split's row count.
Each manifest has the split's source file, its ids in source file order and the sha256 of its rows' Example.to_json()
lines, so load_split notices a changed source. No timestamps, so a rerun is byte-identical.
"""

import hashlib
import json
from pathlib import Path

from thinkguard.data.audit import TRAIN_FILTERED
from thinkguard.data.normalize import INTERIM
from thinkguard.data.schema import read_examples
from thinkguard.data.sources import TEST_SETS
from thinkguard.data.split import LONGEST, PILOT_SIZE, SMOKE_SIZE, choose_seed, group_split, subset

MANIFESTS = Path("data/manifests")


def sha256(examples):
    """Hash of the rows' Example.to_json() lines joined with newlines: row content, not file bytes."""
    return hashlib.sha256("\n".join(e.to_json() for e in examples).encode("utf-8")).hexdigest()


def write(split, source, examples, manifests=MANIFESTS):
    manifest = {
        "split": split,
        "source": source.as_posix(),
        "ids": [e.example_id for e in examples],
        "sha256": sha256(examples),
    }
    manifests.mkdir(parents=True, exist_ok=True)
    with open(manifests / f"{split}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")


def load_split(name, manifests=MANIFESTS):
    """The split's Examples in manifest order. Raises ValueError if an id is not in its source or the rows' sha256
    differs from the manifest.
    """
    with open(manifests / f"{name}.json", encoding="utf-8") as f:
        manifest = json.load(f)
    rows = {e.example_id: e for e in read_examples(manifest["source"])}
    missing = [i for i in manifest["ids"] if i not in rows]
    if missing:
        raise ValueError(f"split {name}: {len(missing)} ids not in {manifest['source']}, first {missing[0]}")
    examples = [rows[i] for i in manifest["ids"]]
    if sha256(examples) != manifest["sha256"]:
        raise ValueError(f"split {name}: sha256 of its rows differs from the manifest")
    return examples


def main():
    kept = [e for e in read_examples(TRAIN_FILTERED) if e.exclusion_reason is None]
    train, validation = group_split(kept, seed=choose_seed(kept)[0])
    splits = {
        "train": (TRAIN_FILTERED, train),
        "validation": (TRAIN_FILTERED, validation),
        "smoke": (TRAIN_FILTERED, subset(train, SMOKE_SIZE)),
        "pilot": (TRAIN_FILTERED, subset(train, PILOT_SIZE, LONGEST)),
    }
    for name in TEST_SETS:
        path = INTERIM / f"{name}.jsonl"
        splits[f"test_{name}"] = path, read_examples(path)
    for split, (source, examples) in splits.items():
        write(split, source, examples)
        print(f"{split}: {len(examples)} rows")


if __name__ == "__main__":
    main()
