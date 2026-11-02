"""thinkguard.config on schemas and YAML files built here."""

import dataclasses
from dataclasses import dataclass

import pytest

from thinkguard.config import config_hash, load


@dataclass(frozen=True)
class Optim:
    lr: float
    warmup: float = 0.03


@dataclass(frozen=True)
class Toy:
    name: str
    seed: int
    optim: Optim


def write(tmp_path, text):
    path = tmp_path / "toy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_reads_nested_fields_and_fills_defaults(tmp_path):
    config = load(write(tmp_path, "name: toy\nseed: 1\noptim:\n  lr: 2.0e-5\n"), Toy)
    assert config == Toy("toy", 1, Optim(2e-5, 0.03))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 2


@pytest.mark.parametrize(
    "text, match",
    [
        ("name: toy\nseed: 1\noptim: {lr: 1.0}\nsede: 2\n", r"unknown keys \['sede'\]"),
        ("name: toy\noptim: {lr: 1.0}\n", r"missing keys \['seed'\]"),
        ("name: toy\nseed: 1\noptim: {lr: 1.0, warmpu: 0.1}\n", r"Optim: unknown keys \['warmpu'\]"),
    ],
)
def test_load_rejects_unknown_and_missing_keys(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load(write(tmp_path, text), Toy)


def test_load_rejects_a_non_mapping_for_a_dataclass_field(tmp_path):
    with pytest.raises(TypeError, match="Optim: expected a mapping"):
        load(write(tmp_path, "name: toy\nseed: 1\noptim: 1.0\n"), Toy)


def test_config_hash_follows_values_not_key_order_or_defaults(tmp_path):
    a = load(write(tmp_path, "name: toy\nseed: 1\noptim: {lr: 1.0}\n"), Toy)
    b = load(write(tmp_path, "optim: {warmup: 0.03, lr: 1.0}\nseed: 1\nname: toy\n"), Toy)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(dataclasses.replace(a, seed=2))
