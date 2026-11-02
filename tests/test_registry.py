"""thinkguard.registry on the committed registry and on registries built here."""

import json

import pytest
import yaml

from thinkguard.registry import load


def entry(id, **fields):
    return {"id": id, "role": "r", "base": "m", "parent": None, "adapter": None, "decode": "greedy"} | fields


def test_committed_registry_has_the_reproduction_conditions():
    entries = load()
    assert list(entries) == ["B", "L", "C", "R"]
    assert entries["L"].parent == entries["C"].parent == "B"
    assert {e.decode for e in entries.values()} == {"greedy"}


@pytest.mark.parametrize(
    "entries, match",
    [
        ([entry("B"), entry("B")], r"repeated ids \['B'\]"),
        ([entry("L", parent="X")], "L: parent 'X' is not an entry"),
        ([entry("A", parent="B"), entry("B", parent="A")], "A: parents form a cycle"),
        ([entry("B", adapter="ckpt/B")], "B: an adapter needs a parent"),
        ([entry("B", base="paper")], "B: base 'paper' is not a model"),
        ([entry("B", decode="sample")], "B: decode 'sample'"),
        ([entry("B") | {"parnet": None}], r"unknown keys \['parnet'\]"),
    ],
)
def test_load_rejects_an_invalid_registry(tmp_path, entries, match):
    (tmp_path / "registry.yaml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    sources = [{"name": "m", "kind": "model"}, {"name": "paper", "kind": "paper"}]
    (tmp_path / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load(tmp_path / "registry.yaml", tmp_path / "sources.json")
