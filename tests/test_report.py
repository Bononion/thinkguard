"""thinkguard.report.score_run on four hand-made rows: matching by id, which metrics it adds, what it refuses."""

from dataclasses import replace

import pytest
from test_prompts import example

from thinkguard.report import score_run

UNSAFE = "- Safety Assessment: unsafe"
# example_id, gold label, gold categories, generated text
ROWS = [
    ("a", "unsafe", ("self_harm",), f"{UNSAFE}\n- Categories: self_harm"),  # true positive
    ("b", "unsafe", ("animal_abuse",), "unsafe\nS1"),  # invalid: a false negative
    ("c", "safe", None, "- Safety Assessment: safe"),  # true negative
    ("d", "safe", None, UNSAFE),  # false positive
]
EXAMPLES = [replace(example(), example_id=i, gold_safety_label=g, gold_categories=c) for i, g, c, _ in ROWS]


def run(split):
    """(manifest, predictions) of a completed run, predictions in the reverse of the split's order: matched by
    position, unsafe_f1 would be 0.4 and no gold-unsafe row invalid.
    """
    manifest = {"run_id": "r1", "status": "completed", "config_hash": "h",
                "config": {"condition": "R", "split": split, "mode": "decision"}}  # fmt: skip
    predictions = [{"example_id": i, "config_hash": "h", "text": t, "unsafe_score": 0.0} for i, _, _, t in ROWS[::-1]]
    return manifest, predictions


@pytest.mark.parametrize("split", ["test_beavertails_30k_test", "test_toxicchat_0124_test"])
def test_score_run(split):
    s = score_run(*run(split), EXAMPLES)
    assert (s["run_id"], s["split"], s["mode"]) == ("r1", split, "decision")
    assert s["unsafe_f1"] == 0.5 and s["confusion.unsafe.invalid"] == 1  # tp 1, fp 1, fn 1
    assert ("category_macro_f1" in s) == (split == "test_beavertails_30k_test")


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda m, p: m.update(status="interrupted"), "not completed"),
        (lambda m, p: p.pop(), "ids differ"),
        (lambda m, p: p.append(p[0]), "ids differ"),
        (lambda m, p: p[1].update(config_hash="other"), "config_hash"),
    ],
)
def test_score_run_rejects(change, match):
    manifest, predictions = run("smoke")
    change(manifest, predictions)
    with pytest.raises(ValueError, match=match):
        score_run(manifest, predictions, EXAMPLES)
