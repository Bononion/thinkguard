"""thinkguard.report on four hand-made rows: score_run (matching by id, which metrics, what it refuses) and the
scores command (byte-identical CSV, one run per condition, split and mode, no torch).
"""

import csv
import io
import json
import subprocess
import sys
from dataclasses import replace

import pytest
from test_prompts import example

from thinkguard import report
from thinkguard.report import score_run

UNSAFE = "- Safety Assessment: unsafe"
# example_id, gold label, gold categories, generated text
ROWS = [
    ("a", "unsafe", ("self_harm", "animal_abuse"), f"{UNSAFE}\n- Categories: self_harm"),  # true positive, overlap
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
    if split == "test_beavertails_30k_test":  # a is wrong under exact (readings 1, 2), right under overlap (3, 4)
        got = {k: v for k, v in s.items() if k.startswith("paper.")}
        assert got == {f"paper.{r}.{k}": v for r in (1, 2, 3, 4) for k, v in zip(("f1", "accuracy"),
                       (0.0, 0.25) if r < 3 else (0.5, 0.5), strict=True)}  # fmt: skip


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


@pytest.fixture
def scores(tmp_path, monkeypatch):
    """report.main with runs and results in tmp_path and EXAMPLES for every split; returns (main, write_run)."""
    monkeypatch.setattr(report, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "load_split", lambda name: EXAMPLES)

    def write_run(run_id, split):
        manifest, predictions = run(split)
        d = tmp_path / "runs" / run_id
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(manifest | {"run_id": run_id}), encoding="utf-8")
        (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")

    return lambda *ids: report.main(["scores", "--runs", *ids]), write_run


def test_scores_csv_is_the_same_bytes_whatever_the_order_of_runs(scores, tmp_path):
    main, write_run = scores
    write_run("r1", "test_beavertails_30k_test")
    write_run("r2", "test_toxicchat_0124_test")
    main("r1", "r2")
    first = (tmp_path / "scores.csv").read_bytes()
    main("r2", "r1")
    assert (tmp_path / "scores.csv").read_bytes() == first
    rows = list(csv.DictReader(io.StringIO(first.decode("utf-8"))))
    assert {(r["run_id"], r["config_hash"], r["value"]) for r in rows if r["metric"] == "unsafe_f1"} == {
        ("r1", "h", "0.5"),
        ("r2", "h", "0.5"),
    }


def test_scores_refuses_two_runs_of_one_condition_split_and_mode(scores, tmp_path):
    main, write_run = scores
    write_run("r1", "smoke")
    write_run("r2", "smoke")
    with pytest.raises(ValueError, match="more than one run"):
        main("r1", "r2")
    assert not (tmp_path / "scores.csv").exists()


def test_report_does_not_import_torch():  # so the tables can be rebuilt without the model weights
    check = "import sys, thinkguard.report; assert 'torch' not in sys.modules, 'torch imported'"
    subprocess.run([sys.executable, "-c", check], check=True)
