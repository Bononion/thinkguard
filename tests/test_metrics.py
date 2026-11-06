"""thinkguard.evaluation.metrics on rows whose metrics are worked out by hand in the comments."""

import pytest

from thinkguard.evaluation.metrics import category_metrics, flag_rates, safety_metrics

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


A, B = "self_harm", "privacy_violation"
CATEGORY_ROWS = [  # (gold, gold_categories, predicted, predicted_categories)
    ("unsafe", {A}, "unsafe", {A}),  # A TP; exact
    ("unsafe", {A, B}, "unsafe", {A}),  # A TP, B FN
    ("safe", set(), "unsafe", {B}),  # B FP: a gold-safe row
    ("unsafe", {B}, None, {B}),  # B FN: an invalid output predicts no category
    ("unsafe", {A}, "safe", {A}),  # A FN: a safe output predicts no category
    (None, None, "unsafe", {A}),  # not scored
    ("unsafe", {A}, "unsafe", {A, "S1"}),  # A TP; not exact, "S1" is not a key
]


def test_category_metrics_by_hand():
    m = category_metrics(CATEGORY_ROWS)
    # A: TP 3, FP 0, FN 1, F1 6/7, support 4. B: TP 0, FP 1, FN 2, F1 0, support 2. The other 12 keys: 0.
    assert m["per_category"][A] == {"f1": pytest.approx(6 / 7), "support": 4}
    assert m["per_category"][B] == {"f1": 0.0, "support": 2}
    assert m["category_macro_f1"] == pytest.approx((6 / 7) / 14)  # 3/49
    assert m["category_exact_match"] == 1 / 5  # 5 gold-unsafe rows, only the first matches exactly


def test_flag_rates_over_scored_rows():
    rows = [("unsafe", {"invalid_label"}), ("safe", {"truncated", "contradiction"}), ("safe", set()), ("unsafe", set())]
    rows.append((None, {"invalid_label", "truncated"}))  # not scored
    assert flag_rates(rows) == {
        "invalid_label_rate": 1 / 4,
        "unknown_category_rate": 0.0,
        "contradiction_rate": 1 / 4,
        "truncated_rate": 1 / 4,
    }
