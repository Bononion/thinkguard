"""Run configs: a YAML file read into a frozen dataclass, and the config's hash.

A schema is a frozen dataclass. Every field without a default must be in the file and an unknown key raises, so a
misspelled key fails at load instead of silently falling back to a default. A field whose type is a dataclass is read
from a nested mapping by the same rules.
"""

import dataclasses
import hashlib
import json
import typing

import yaml


def from_dict(schema, d):
    """schema built from the mapping d. Raises TypeError if d is not a mapping, ValueError if it has a key that is not
    a field or lacks a field that has no default.
    """
    if not isinstance(d, dict):
        raise TypeError(f"{schema.__name__}: expected a mapping, got {type(d).__name__}")
    fields = dataclasses.fields(schema)
    unknown = sorted(set(d) - {f.name for f in fields})
    missing = [
        f.name
        for f in fields
        if f.name not in d and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    if unknown or missing:
        raise ValueError(f"{schema.__name__}: unknown keys {unknown}, missing keys {missing}")
    types = typing.get_type_hints(schema)
    return schema(**{k: from_dict(types[k], v) if dataclasses.is_dataclass(types[k]) else v for k, v in d.items()})


def load(path, schema):
    """The YAML file at path as a schema instance, by the rules of from_dict. An empty file is an empty mapping."""
    with open(path, encoding="utf-8") as f:
        return from_dict(schema, yaml.safe_load(f) or {})


def config_hash(config):
    """sha256 of the config's canonical JSON: every field, defaults included, keys sorted, no whitespace."""
    canonical = json.dumps(dataclasses.asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
