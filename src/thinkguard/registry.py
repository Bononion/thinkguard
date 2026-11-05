"""The experiment registry (configs/experiment_registry.yaml): one entry per condition in protocol.md, Conditions.

An entry names its base model by its name in configs/frozen/source_manifest.json, which pins the revision, and the
entry it is trained from (parent). adapter is the path of the trained LoRA adapter, null until the entry is trained.
The file is a list, not a mapping keyed by id, so a repeated id is an error instead of YAML keeping only the last one.
"""

import collections
import json
from dataclasses import dataclass

import yaml

from thinkguard.config import from_dict
from thinkguard.paths import ROOT

REGISTRY = ROOT / "configs/experiment_registry.yaml"
SOURCE_MANIFEST = ROOT / "configs/frozen/source_manifest.json"
DECODES = ("greedy", "self_speculative")
HF = "https://huggingface.co/"


@dataclass(frozen=True)
class Entry:
    id: str
    role: str
    base: str
    parent: str | None
    adapter: str | None
    decode: str


def hub(name, sources=SOURCE_MANIFEST):
    """(repo_id, revision) of the model called name in sources, for from_pretrained. Raises ValueError if name is not
    a Hugging Face model there.
    """
    with open(sources, encoding="utf-8") as f:
        source = next((s for s in json.load(f) if s["name"] == name and s["kind"] == "model"), None)
    if source is None or not source["url"].startswith(HF):
        raise ValueError(f"{name!r} is not a Hugging Face model in {sources}")
    return source["url"].removeprefix(HF), source["revision"]


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
