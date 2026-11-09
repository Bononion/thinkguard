"""Results tables from stored predictions, never from re-generation (#54). Needs no model weights and no torch.

Run from the repo root: python -m thinkguard.report scores --runs <id> [<id> ...]
Reads $THINKGUARD_RUNS/<id>/manifest.json and predictions.jsonl, and the split's rows (load_split). Writes
results/scores.csv: one row per run and metric, with the run's id and config hash. Human-readable tables (gate.md,
the report's tables) are built from it.
"""

import argparse
import csv
import json
from collections import Counter

from thinkguard.data.manifest import load_split
from thinkguard.evaluation.metrics import category_metrics, flag_rates, safety_metrics
from thinkguard.evaluation.parse import parse
from thinkguard.paths import ROOT
from thinkguard.runs import RUNS

RESULTS = ROOT / "results"
CATEGORY_SPLIT = "test_beavertails_30k_test"  # protocol.md, Categories (BeaverTails test only)
META = ("run_id", "config_hash", "condition", "split", "mode")


def flat(d, prefix=""):
    """d with nested dicts turned into dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    out = {}
    for k, v in d.items():
        out |= flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def score_run(manifest, predictions, examples):
    """{run_id, config_hash, condition, split, mode, then the metrics of protocol.md, Metrics, with nested ones as
    dotted keys (confusion.unsafe.invalid, per_category.<key>.f1)} for one run. predictions are its predictions.jsonl
    rows, examples its split's Examples; a row's gold comes from the Example with its example_id. Category metrics
    only on CATEGORY_SPLIT. Raises ValueError if the run did not complete, the predictions' example ids are not the
    split's (missing, extra or repeated), or a row has another config_hash than the manifest.
    """
    run_id, h, config = manifest["run_id"], manifest["config_hash"], manifest["config"]
    if manifest["status"] != "completed":
        raise ValueError(f"{run_id}: status {manifest['status']}, not completed")
    got, want = Counter(p["example_id"] for p in predictions), Counter(e.example_id for e in examples)
    if got != want:
        raise ValueError(
            f"{run_id}: example ids differ from the split's, first {sorted((got - want) + (want - got))[:5]}"
        )
    if any(p["config_hash"] != h for p in predictions):
        raise ValueError(f"{run_id}: a prediction has another config_hash than the manifest")
    predicted = {p["example_id"]: (parse(p["text"]), p["unsafe_score"]) for p in predictions}
    rows = [(e, *predicted[e.example_id]) for e in examples]
    out = {"run_id": run_id, "config_hash": h, **{k: config[k] for k in ("condition", "split", "mode")}}
    out |= flat(safety_metrics((e.gold_safety_label, p.label, s) for e, p, s in rows))
    out |= flag_rates((e.gold_safety_label, p.flags) for e, p, _ in rows)
    if config["split"] == CATEGORY_SPLIT:
        out |= flat(
            category_metrics((e.gold_safety_label, e.gold_categories, p.label, p.categories) for e, p, _ in rows)
        )
    return out


def write_scores(scores, path):
    """Writes path, a CSV with columns META + (metric, value): one row per metric of each score_run result, sorted by
    (split, condition, mode, metric), so the order of scores does not matter. Raises ValueError, before writing, if
    two results share (condition, split, mode).
    """
    seen = Counter((s["condition"], s["split"], s["mode"]) for s in scores)
    repeated = sorted(k for k, n in seen.items() if n > 1)
    if repeated:
        raise ValueError(f"more than one run for (condition, split, mode) {repeated}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow((*META, "metric", "value"))
        for s in sorted(scores, key=lambda s: (s["split"], s["condition"], s["mode"])):
            out.writerows((*(s[k] for k in META), m, s[m]) for m in sorted(set(s) - set(META)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Results tables from stored predictions.")
    scores = parser.add_subparsers(dest="table", required=True).add_parser("scores", help="results/scores.csv")
    scores.add_argument("--runs", nargs="+", required=True, help="run ids under THINKGUARD_RUNS")
    args = parser.parse_args(argv)
    results = []
    for run_id in args.runs:
        manifest = json.loads((RUNS / run_id / "manifest.json").read_text(encoding="utf-8"))
        lines = (RUNS / run_id / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        results.append(score_run(manifest, [json.loads(x) for x in lines], load_split(manifest["config"]["split"])))
    RESULTS.mkdir(exist_ok=True)
    write_scores(results, RESULTS / "scores.csv")


if __name__ == "__main__":
    main()
