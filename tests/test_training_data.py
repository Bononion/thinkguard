"""thinkguard.training.data: which tokens are supervised in L and C, padding, and the max_length check."""

import pytest
from test_prompts import example

torch = pytest.importorskip("torch")

from thinkguard.prompts import encode  # noqa: E402
from thinkguard.training.data import IGNORE, collate, rows  # noqa: E402


def supervised(row):
    return [k for k, label in enumerate(row["labels"]) if label != IGNORE]


def test_l_supervises_the_decision_and_eos_and_c_adds_the_explanation():
    e = example()
    (l_row,), (c_row,) = rows([e], "train_label_category", 2048), rows([e], "train_critique", 2048)
    l_spans, c_spans = encode(e, "train_label_category").spans, encode(e, "train_critique").spans
    assert supervised(l_row) == list(range(l_spans["decision"][0], l_spans["eos"][1]))  # no explanation token
    assert supervised(c_row) == list(range(c_spans["decision"][0], c_spans["eos"][1]))
    a, b = c_spans["decision"]
    assert c_row["labels"][a:b] == l_row["labels"][a:b] == c_row["input_ids"][a:b]  # C's decision tokens are L's
    assert l_row["length"] == c_row["length"] == len(c_row["input_ids"])  # grouped by C's length


def test_collate_pads_on_the_right_without_supervising_padding():
    short, long = rows([example(None), example()], "train_critique", 2048)
    batch = collate([short, long])
    n, p = len(long["input_ids"]), len(long["input_ids"]) - len(short["input_ids"])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape == (2, n)
    assert batch["labels"][0].tolist() == short["labels"] + [IGNORE] * p
    assert batch["attention_mask"][0].tolist() == [1] * (n - p) + [0] * p
    assert batch["input_ids"][1].tolist() == long["input_ids"]


def test_rows_raise_on_a_row_longer_than_max_length():
    n = len(encode(example(), "train_critique").input_ids)
    assert rows([example()], "train_critique", n)  # exactly max_length is kept
    with pytest.raises(ValueError, match="max_length"):
        rows([example()], "train_critique", n - 1)
