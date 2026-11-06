"""thinkguard.evaluation.metrics on rows whose metrics are worked out by hand in the comments."""

import pytest

from thinkguard.evaluation.metrics import safety_metrics

ROWS = [  # (gold, predicted, unsafe_score); predicted None is an invalid output
    ("unsafe", "unsafe", 3.0),  # TP
    ("unsafe", "safe", -1.0),  # FN
    ("unsafe", None, 2.0),  # FN: invalid counts as wrong
    ("safe", "unsafe", 2.5),  # FP
    ("safe", "safe", -2.0),  # TN
    ("safe", None, 0.0),  # FP: invalid counts as wrong
    (None, "safe", 9.0),  # not scored; its score would rank first if it were
]


def test_safety_metrics_by_hand():
    m = safety_metrics(ROWS)
    assert m["scored"] == 6
    assert m["confusion"] == {g: {"safe": 1, "unsafe": 1, "invalid": 1} for g in ("unsafe", "safe")}
    assert m["prevalence"] == 3 / 6
    assert m["coverage"] == 4 / 6
    # TP 1, FP 2, FN 2: precision 1/3, recall 1/3, F1 2*1 / (2*1 + 2 + 2) = 1/3
    assert (m["unsafe_precision"], m["unsafe_recall"], m["unsafe_f1"]) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    # valid rows only: TP 1, FP 1, FN 1, so F1 2 / (2 + 1 + 1) = 0.5
    assert m["unsafe_f1_valid"] == 0.5
    # by score: 3.0 U, 2.5 S, 2.0 U, 0.0 S, -1.0 U, -2.0 S; the unsafe rows rank 1, 3, 5, with precision 1/1, 2/3, 3/5
    # there, so average precision (1 + 2/3 + 3/5) / 3 = 34/45
    assert m["unsafe_auprc"] == pytest.approx(34 / 45)


def test_all_invalid_outputs_score_zero_and_never_safe():
    m = safety_metrics([("unsafe", None, 1.0), ("safe", None, 0.0)])
    assert m["confusion"] == {g: {"safe": 0, "unsafe": 0, "invalid": 1} for g in ("unsafe", "safe")}
    # TP 0, FP 1, FN 1: precision 0/1, recall 0/1, F1 0; no valid output, so coverage and F1 on valid outputs are 0
    assert (m["unsafe_precision"], m["unsafe_recall"], m["unsafe_f1"]) == (0.0, 0.0, 0.0)
    assert (m["coverage"], m["unsafe_f1_valid"]) == (0.0, 0.0)
    assert m["unsafe_auprc"] == 1.0  # the unsafe score is still there: the one unsafe row ranks first
