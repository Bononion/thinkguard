"""thinkguard.training.config: the committed train configs load, L and C differ only in mode, bad values raise."""

import dataclasses

import pytest

from thinkguard.config import load
from thinkguard.paths import ROOT
from thinkguard.training.config import TrainConfig

TRAIN = ROOT / "configs/train"


@pytest.mark.parametrize("name", ["", "overfit_", "pilot_"])
def test_l_and_c_configs_differ_only_in_mode(name):
    l_config = dataclasses.asdict(load(TRAIN / f"{name}label_category.yaml", TrainConfig))
    c_config = dataclasses.asdict(load(TRAIN / f"{name}critique.yaml", TrainConfig))
    assert {k for k in l_config if l_config[k] != c_config[k]} == {"mode"}


@pytest.mark.parametrize(
    "change, match",
    [({"mode": "infer"}, "mode"), ({"eval_split": "test_wildguardtest"}, "test split"), ({"lr": "2e-4"}, "float"),
     ({"grad_accum": 0}, "positive")],
)  # fmt: skip
def test_train_config_rejects(change, match):
    good = dataclasses.asdict(load(TRAIN / "critique.yaml", TrainConfig))
    with pytest.raises(ValueError, match=match):
        TrainConfig(**good | change)
