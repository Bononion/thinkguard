"""Score label-only predictions: F1, AUPRC, accuracy and macro F1 (0-100) with bootstrap 95% CIs.

Run from the repo root inside .venv:
    python eval/metrics.py --tag zero_shot                  # score results/preds/zero_shot/*.jsonl into results/scores.csv
    python eval/metrics.py --diff A.jsonl B.jsonl           # paired bootstrap of B - A per metric; prints, writes nothing
    python eval/metrics.py --selftest
Unparsed rows count as wrong: a null pred becomes the opposite of the label, and a null p_unsafe becomes 0.0 on an
unsafe row and 1.0 on a safe row. CIs are percentiles 2.5/97.5 over 1,000 row resamples and are for reporting only.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

SEED = 42
N_BOOT = 1000
METRICS = ["f1", "auprc", "accuracy", "macro_f1"]
COLUMNS = ["tag", "benchmark", "prompt", "mode", "metric", "value", "ci_low", "ci_high"]


def arrays(rows):
    """Labels, filled preds and filled p_unsafe as arrays, plus the null count of each field."""
    y = np.array([r["label"] == "unsafe" for r in rows])
    pred = np.array([r["pred"] == "unsafe" if r["pred"] is not None else not u for r, u in zip(rows, y)])
    p = np.array([r["p_unsafe"] if r["p_unsafe"] is not None else float(not u) for r, u in zip(rows, y)])
    counts = {"unparsed_pred": sum(r["pred"] is None for r in rows),
              "unparsed_p_unsafe": sum(r["p_unsafe"] is None for r in rows)}
    return y, pred, p, counts


def load(path):
    """Read a preds file, check it against data/test, and return its key and arrays in data/test row order."""
    name = os.path.basename(path)
    parts = name.removesuffix(".jsonl").split("__")
    if len(parts) != 3:
        sys.exit(f"{path}: expected a name like <benchmark>__<prompt>__<mode>.jsonl")
    benchmark, prompt, mode = parts
    test = [json.loads(line) for line in open(f"data/test/{benchmark}.jsonl", encoding="utf-8")]
    preds = [json.loads(line) for line in open(path, encoding="utf-8")]
    by_id = {r["id"]: r for r in preds}
    test_ids = {r["id"] for r in test}
    missing, extra, dups = len(test_ids - by_id.keys()), len(by_id.keys() - test_ids), len(preds) - len(by_id)
    if missing or extra or dups:
        sys.exit(f"{path}: {len(preds)} rows, data/test/{benchmark}.jsonl has {len(test)}; "
                 f"{missing} ids missing, {extra} extra, {dups} duplicate")
    rows = [by_id[r["id"]] for r in test]
    mismatched = sum(r["label"] != t["label"] for r, t in zip(rows, test))
    if mismatched:
        sys.exit(f"{path}: {mismatched} of {len(rows)} labels differ from data/test/{benchmark}.jsonl")
    return {"benchmark": benchmark, "prompt": prompt, "mode": mode}, *arrays(rows)


def metrics(y, pred, p, idx):
    """Every metric (0-100) on each row set in idx, shape (k, n); returns arrays of shape (k,)."""
    yy, pp = y[idx], pred[idx]
    tp, fp = (yy & pp).sum(1), (~yy & pp).sum(1)
    fn, tn = (yy & ~pp).sum(1), (~yy & ~pp).sum(1)
    # F1 = 2tp / (2tp + fp + fn), 0 when nothing is positive (zero_division=0); safe-F1 swaps the classes
    f1_unsafe = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(idx)), where=2 * tp + fp + fn > 0)
    f1_safe = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros(len(idx)), where=2 * tn + fn + fp > 0)
    return {"f1": 100 * f1_unsafe,
            "auprc": 100 * np.array([average_precision_score(y[i], p[i]) for i in idx]),
            "accuracy": 100 * (tp + tn) / idx.shape[1],
            "macro_f1": 50 * (f1_unsafe + f1_safe)}


def point(y, pred, p):
    return {m: v[0] for m, v in metrics(y, pred, p, np.arange(len(y))[None]).items()}


def bootstrap(y, pred, p):
    # A fresh generator per file: files on the same benchmark (same row order) get the same resamples
    idx = np.random.default_rng(SEED).integers(0, len(y), (N_BOOT, len(y)))
    return metrics(y, pred, p, idx)


def ci(values):
    return np.percentile(values, [2.5, 97.5])


def score_tag(tag, results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "preds", tag, "*.jsonl")))
    if not files:
        sys.exit(f"no preds files in {os.path.join(results_dir, 'preds', tag)}")
    new = []
    for path in files:  # every file is checked before scores.csv is touched
        key, y, pred, p, counts = load(path)
        key = {"tag": tag, **key}
        value, boot = point(y, pred, p), bootstrap(y, pred, p)
        for m in METRICS:
            lo, hi = ci(boot[m])
            new.append({**key, "metric": m, "value": f"{value[m]:.1f}", "ci_low": f"{lo:.1f}", "ci_high": f"{hi:.1f}"})
        new += [{**key, "metric": m, "value": str(c), "ci_low": "", "ci_high": ""} for m, c in counts.items()]

    out = os.path.join(results_dir, "scores.csv")
    old = []
    if os.path.exists(out):
        with open(out, encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if r["tag"] != tag]
    rows = sorted(old + new, key=lambda r: [r[c] for c in COLUMNS[:5]])
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(new)} rows for tag {tag} from {len(files)} files")


def diff(path_a, path_b):
    (key_a, y, pred_a, p_a, _), (key_b, _, pred_b, p_b, _) = load(path_a), load(path_b)
    # Both match data/test, so ids and labels agree exactly when the benchmark does
    if key_a["benchmark"] != key_b["benchmark"]:
        sys.exit(f"{path_a} and {path_b} are on different benchmarks, so their ids and labels differ")
    va, vb = point(y, pred_a, p_a), point(y, pred_b, p_b)
    ba, bb = bootstrap(y, pred_a, p_a), bootstrap(y, pred_b, p_b)  # same seed, same n: paired resamples
    print(f"A: {path_a}\nB: {path_b}")
    print(f"{'metric':<10}{'A':>8}{'B':>8}{'B-A':>8}{'ci_low':>8}{'ci_high':>8}")
    for m in METRICS:
        lo, hi = ci(bb[m] - ba[m])
        print(f"{m:<10}{va[m]:8.1f}{vb[m]:8.1f}{vb[m] - va[m]:8.1f}{lo:8.1f}{hi:8.1f}")


def selftest():
    from sklearn.metrics import accuracy_score, f1_score

    toy = [  # label, pred, p_unsafe -> filled pred, filled p_unsafe
        {"label": "unsafe", "pred": "unsafe", "p_unsafe": 0.9},  # TP, 0.9
        {"label": "unsafe", "pred": "safe", "p_unsafe": 0.6},    # FN, 0.6
        {"label": "unsafe", "pred": "safe", "p_unsafe": None},   # FN, null p_unsafe on an unsafe row -> 0.0
        {"label": "safe", "pred": "safe", "p_unsafe": 0.2},      # TN, 0.2
        {"label": "safe", "pred": None, "p_unsafe": 0.7},        # null pred on a safe row -> unsafe: FP, 0.7
        {"label": "safe", "pred": "safe", "p_unsafe": 0.4},      # TN, 0.4
    ]
    y, pred, p, counts = arrays(toy)
    assert counts == {"unparsed_pred": 1, "unparsed_p_unsafe": 1}
    assert pred.tolist() == [True, False, False, False, True, False] and p.tolist() == [0.9, 0.6, 0.0, 0.2, 0.7, 0.4]
    got = point(y, pred, p)
    # TP 1, FN 2, FP 1, TN 2
    # f1 = 2*1 / (2*1 + 1 + 2) = 2/5 = 40.0
    # accuracy = (1 + 2) / 6 = 50.0
    # safe-F1 = 2*2 / (2*2 + 2 + 1) = 4/7; macro_f1 = (2/5 + 4/7) / 2 = 34/70 = 48.571...
    # auprc: by p desc 0.9(u) 0.7(s) 0.6(u) 0.4(s) 0.2(s) 0.0(u); recall steps at 0.9, 0.6, 0.0 with
    #   precision 1, 2/3, 3/6 -> (1 + 2/3 + 1/2) / 3 = 13/18 = 72.222...
    expected = {"f1": 40.0, "accuracy": 50.0, "macro_f1": 100 * 34 / 70, "auprc": 100 * 13 / 18}
    for m, v in expected.items():
        assert np.isclose(got[m], v), f"toy {m}: {got[m]} != {v}"
    # The other direction of each rule: null p_unsafe on a safe row -> 1.0, null pred on an unsafe row -> safe
    _, pred2, p2, _ = arrays([{"label": "safe", "pred": "safe", "p_unsafe": None},
                              {"label": "unsafe", "pred": None, "p_unsafe": 0.5}])
    assert p2.tolist() == [1.0, 0.5] and pred2.tolist() == [False, False]

    rng = np.random.default_rng(0)
    random_rows = (rng.random(200) < 0.4, rng.random(200) < 0.5, rng.random(200))
    for name, (y, pred, p) in [("toy", (y, pred, p)), ("200 random rows", random_rows)]:
        got = point(y, pred, p)
        sk = {"f1": f1_score(y, pred, zero_division=0),
              "accuracy": accuracy_score(y, pred),
              "macro_f1": f1_score(y, pred, average="macro", labels=[False, True], zero_division=0)}
        for m, v in sk.items():
            assert np.isclose(got[m], 100 * v), f"{name} {m}: numpy {got[m]} != sklearn {100 * v}"
    print("selftest passed: toy f1 40.0, accuracy 50.0, macro_f1 48.6, auprc 72.2, 1 null pred, 1 null p_unsafe; "
          "numpy f1/accuracy/macro_f1 == sklearn on the toy and 200 random rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="score results/preds/<tag>/*.jsonl into scores.csv, replacing that tag's rows")
    g.add_argument("--diff", nargs=2, metavar=("A", "B"), help="paired bootstrap of B - A for two preds files")
    g.add_argument("--selftest", action="store_true")
    p.add_argument("--results-dir", default="results", help="root of preds/ and scores.csv")
    args = p.parse_args()
    if args.selftest:
        selftest()
    elif args.diff:
        diff(*args.diff)
    else:
        score_tag(args.tag, args.results_dir)
