"""thinkguard.evaluation.bootstrap: the observed difference agrees with metrics, identical conditions give 0, and
resampling draws whole clusters (a two-cluster case whose every resample is worked out by hand)."""

import random

import numpy as np
import pytest

from thinkguard.evaluation.bootstrap import paired_diff
from thinkguard.evaluation.metrics import safety_metrics


def random_rows(n, seed=0):
    rng = random.Random(seed)
    labels = ["safe", "unsafe", None]
    return [(rng.randrange(n // 3), rng.choice(labels[:2]), rng.choice(labels), rng.choice(labels)) for _ in range(n)]


def test_diff_is_the_difference_of_safety_metrics_unsafe_f1():
    rows = random_rows(300)
    diff, low, high, diffs = paired_diff(rows, resamples=1000)
    a = safety_metrics((g, pa, 0.0) for _, g, pa, _ in rows)["unsafe_f1"]
    b = safety_metrics((g, pb, 0.0) for _, g, _, pb in rows)["unsafe_f1"]
    assert diff == pytest.approx(a - b)
    assert len(diffs) == 1000 and low <= high


def test_identical_conditions_differ_by_zero_in_every_resample():
    rows = [(c, g, a, a) for c, g, a, _ in random_rows(300)]
    diff, low, high, diffs = paired_diff(rows, resamples=1000)
    assert (diff, low, high) == (0.0, 0.0, 0.0) and not diffs.any()


def test_resamples_draw_whole_clusters():
    # cluster x: two gold-unsafe rows, a right, b wrong; cluster y: one gold-unsafe row, both right. a's F1 is 1 in
    # every resample. b's: {x, x} tp 0 -> 0; {x, y} tp 1, fn 2 -> 2/4; {y, y} -> 1. So a - b is 1, 0.5 or 0, with
    # probability 1/4, 1/2, 1/4; resampling single rows would also give, e.g., 2/3
    rows = [("x", "unsafe", "unsafe", "safe")] * 2 + [("y", "unsafe", "unsafe", "unsafe")]
    diff, _, _, diffs = paired_diff(rows)
    assert diff == pytest.approx(1 - 2 / 4)
    values, counts = np.unique(diffs, return_counts=True)
    assert values.tolist() == [0.0, 0.5, 1.0]
    assert counts / len(diffs) == pytest.approx([0.25, 0.5, 0.25], abs=0.02)


def test_seed_fixes_the_resamples():
    rows = random_rows(300)
    first, again, other = (paired_diff(rows, resamples=600, seed=s)[3] for s in (0, 0, 1))
    assert (first == again).all() and not (first == other).all()


def test_no_rows_raises():
    with pytest.raises(ValueError, match="no scored rows"):
        paired_diff([])
