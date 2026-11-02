"""thinkguard.evaluation.paper on six hand-scored BeaverTails rows."""

import pytest

from thinkguard.evaluation.paper import beavertails_scores, choose

V, T = "violence", "theft"
ROWS = [
    ("safe", set(), "safe", set()),  # correct
    ("safe", set(), "unsafe", {V}),  # FP
    ("unsafe", {V, T}, "unsafe", {V, T}),  # TP
    ("unsafe", {V, T}, "unsafe", {V}),  # exact: wrong (FN, + FP for fn_fp); overlap: TP
    ("unsafe", {V}, None, set()),  # invalid: FN
    ("unsafe", {V}, "unsafe", {T}),  # wrong under every reading (FN, + FP for fn_fp)
]


def test_scores_follow_the_hand_counts():
    # F1 = 2TP / (2TP + FP + FN); accuracy = correct / 6
    assert beavertails_scores(ROWS) == {
        1: (pytest.approx(100 * 2 / 6), pytest.approx(100 * 2 / 6)),  # TP 1, FP 1, FN 3; correct 2
        2: (pytest.approx(100 * 2 / 8), pytest.approx(100 * 2 / 6)),  # TP 1, FP 3, FN 3
        3: (pytest.approx(100 * 4 / 7), pytest.approx(100 * 3 / 6)),  # TP 2, FP 1, FN 2; correct 3
        4: (pytest.approx(100 * 4 / 8), pytest.approx(100 * 3 / 6)),  # TP 2, FP 2, FN 2
    }


def test_all_invalid_scores_zero():
    scores = beavertails_scores([("unsafe", {V}, None, set()), ("safe", set(), None, set())])
    assert set(scores.values()) == {(0.0, 0.0)}


def test_choose_takes_the_closest_reading_and_the_lower_one_on_a_tie():
    scores = beavertails_scores(ROWS)
    assert choose(scores, f1=50.0, accuracy=50.0) == 4
    assert choose({1: (10.0, 10.0), 2: (10.0, 10.0)}, f1=10.0, accuracy=10.0) == 1
