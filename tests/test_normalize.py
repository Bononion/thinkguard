"""thinkguard.data.normalize on tiny fixture rows (no network, no raw files)."""

from functools import partial

import pytest

from thinkguard.data import normalize as nz

SOURCE = {"revision": "r1", "split": "test"}
KEYS = ["self_harm", "violence,aiding_and_abetting,incitement"]
BT_RULE = {"field": "is_safe", "equals": False}
SAFE_CATS = {"self_harm": False, "violence,aiding_and_abetting,incitement": False}
HARM_CATS = {"self_harm": True, "violence,aiding_and_abetting,incitement": False}

GOLD = nz.gold_annotations(  # the beavertails_30k_train annotations the critique rows are looked up in
    [
        {"prompt": "p0", "response": "r0", "is_safe": False, "category": HARM_CATS},
        {"prompt": "p0", "response": "r0", "is_safe": False, "category": HARM_CATS},  # agrees: still one label
        {"prompt": "p1", "response": "r1", "is_safe": True, "category": SAFE_CATS},
        {"prompt": "p1", "response": "r1", "is_safe": False, "category": HARM_CATS},  # conflicts
    ],
    BT_RULE,
)
UNSAFE = "- Safety Assessment: unsafe\n- Categories: self_harm\n- Explanation: Harmful."
SAFE = "- Safety Assessment: safe\n- Explanation: Fine."

CASES = {  # source -> (to_fields, raw rows, dropped by reason, (gold label, gold categories, status) per written row)
    "critique": (
        partial(nz.critique, keys=KEYS, gold=GOLD),
        [
            {"input": "User: p0\nAgent: r0", "output": UNSAFE},
            {"input": "User: p1\nAgent: r1", "output": SAFE},
            {"input": "User: p2\nAgent: r2", "output": UNSAFE},  # not in BeaverTails
            {"input": "p3 with no User/Agent turns", "output": SAFE},
        ],
        {"input_unparsed": 1},
        [("unsafe", ("self_harm",), "human"), (None, None, "conflicting"), (None, None, "missing")],
    ),
    "wildguardtest": (
        partial(nz.wildguardtest, rule={"field": "response_harm_label", "equals": "harmful"}),
        [
            {"prompt": "p", "response": "r", "response_harm_label": "harmful"},
            {"prompt": "p", "response": "r", "response_harm_label": None},
            {"prompt": "p", "response": None, "response_harm_label": "unharmful"},
        ],
        {"response_missing": 1},
        [("unsafe", (), "human"), (None, None, "missing")],
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_written_and_dropped_rows(name):
    to_fields, rows, dropped, gold = CASES[name]
    examples, status = nz.normalize(name, SOURCE, rows, to_fields)
    assert status == {"ok": len(gold), **dropped}
    assert [(e.gold_safety_label, e.gold_categories, e.annotation_status) for e in examples] == gold
    assert all((e.pair_hash is None) == (e.task_type == "prompt_only") for e in examples)


def test_critique_teacher_label_is_kept_when_gold_is_null():
    to_fields, rows, *_ = CASES["critique"]
    examples, _ = nz.normalize("critique", SOURCE, rows, to_fields)
    conflicting, unmatched = examples[1], examples[2]
    assert (conflicting.teacher_safety_label, conflicting.gold_safety_label) == ("safe", None)
    assert conflicting.annotation_status == "conflicting"
    assert (unmatched.teacher_safety_label, unmatched.teacher_categories) == ("unsafe", ("self_harm",))
    assert (unmatched.gold_safety_label, unmatched.annotation_status) == (None, "missing")
