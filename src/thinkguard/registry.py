"""The experiment registry (configs/experiment_registry.yaml): one entry per condition in protocol.md, Conditions.

An entry names its base model by its name in configs/frozen/source_manifest.json, which pins the revision, and the
entry it is trained from (parent). adapter is the path of the trained LoRA adapter, null until the entry is trained.
The file is a list, not a mapping keyed by id, so a repeated id is an error instead of YAML keeping only the last one.
"""

import collections
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from thinkguard.config import from_dict

REGISTRY = Path("configs/experiment_registry.yaml")
SOURCE_MANIFEST = Path("configs/frozen/source_manifest.json")
DECODES = ("greedy", "self_speculative")


@dataclass(frozen=True)
class Entry:
    id: str
    role: str
    base: str
    parent: str | None
    adapter: str | None
    decode: str


def load(path=REGISTRY, sources=SOURCE_MANIFEST):
    """The registry's entries by id, in file order. Raises ValueError if an id repeats, a parent is not an entry or
    parents form a cycle, an entry without a parent has an adapter, a base is not a model in sources, or decode is not
    in DECODES; from_dict raises on an unknown or missing key.
    """
    with open(path, encoding="utf-8") as f:
        entries = [from_dict(Entry, e) for e in yaml.safe_load(f)]
    with open(sources, encoding="utf-8") as f:
        models = {s["name"] for s in json.load(f) if s["kind"] == "model"}
    repeated = sorted(i for i, n in collections.Counter(e.id for e in entries).items() if n > 1)
    if repeated:
        raise ValueError(f"repeated ids {repeated}")
    by_id = {e.id: e for e in entries}
    for e in entries:
        if e.base not in models:
            raise ValueError(f"{e.id}: base {e.base!r} is not a model in {sources}")
        if e.decode not in DECODES:
            raise ValueError(f"{e.id}: decode {e.decode!r} is not one of {DECODES}")
        if e.parent is None and e.adapter is not None:
            raise ValueError(f"{e.id}: an adapter needs a parent to be trained from")
        seen, p = {e.id}, e.parent
        while p is not None:
            if p not in by_id:
                raise ValueError(f"{e.id}: parent {p!r} is not an entry")
            if p in seen:
                raise ValueError(f"{e.id}: parents form a cycle through {p!r}")
            seen.add(p)
            p = by_id[p].parent
    return by_id
