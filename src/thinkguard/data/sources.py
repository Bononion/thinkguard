"""Data sources at pinned revisions (configs/data/sources.yaml), loaded with datasets.load_dataset.

The revision is part of each url, so a load always reads the same files; rows and unsafe_rows in sources.yaml check
them on every load. Files are cached by datasets (HF_HOME). huggingface.co requests send the Hugging Face login token,
which a source marked gated needs.
"""

from pathlib import Path

import yaml
from datasets import load_dataset

SOURCES = Path("configs/data/sources.yaml")
BUILDERS = {"json": "json", "jsonl.gz": "json", "jsonl.xz": "json", "csv": "csv", "parquet": "parquet"}


def load_sources(path=SOURCES):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def unsafe_fields(rule):
    """The gold-label fields of a source's unsafe rule: {field, equals} or {any_of: [fields], equals}."""
    return rule["any_of"] if "any_of" in rule else [rule["field"]]


def is_unsafe(row, rule):
    """True if any of the rule's unsafe_fields equals its value; a missing field never matches."""
    return any(row.get(f) == rule["equals"] for f in unsafe_fields(rule))


def url(source):
    return source["url"].format(revision=source["revision"])


def load(name, path=SOURCES):
    """One source's rows as dicts, in file order. Raises if rows or unsafe_rows differ from sources.yaml.

    A row is unsafe once if any of its unsafe_fields equals the rule's value. split="train" is only the name datasets
    gives a single data file, whatever the source's split.
    """
    source = next(s for s in load_sources(path) if s["name"] == name)
    rows = load_dataset(BUILDERS[source["format"]], data_files=url(source), split="train").to_list()
    unsafe = sum(is_unsafe(r, source["unsafe"]) for r in rows) if "unsafe" in source else None
    if (len(rows), unsafe) != (source["rows"], source.get("unsafe_rows")):
        raise ValueError(
            f"source {name}: {len(rows)} rows, {unsafe} unsafe; sources.yaml has {source['rows']} rows,"
            f" {source.get('unsafe_rows')} unsafe"
        )
    return rows
