"""Results tables from stored predictions, never from re-generation (#54). Needs no model weights and no torch.

Run from the repo root: python -m thinkguard.report scores --runs <id> [<id> ...]
Reads $THINKGUARD_RUNS/<id>/manifest.json and predictions.jsonl, and the split's rows (load_split). Writes
results/scores.csv: one row per run and metric, with the run's id and config hash. Human-readable tables are built
from it: python -m thinkguard.report gate writes results/gate.md (#13) from it and results/paper_targets.csv.
"""

import argparse
import csv
import json
from collections import Counter

from thinkguard.data.manifest import load_split
from thinkguard.evaluation.metrics import category_metrics, flag_rates, safety_metrics
from thinkguard.evaluation.paper import READINGS, beavertails_scores, choose
from thinkguard.evaluation.parse import parse
from thinkguard.paths import ROOT
from thinkguard.runs import RUNS

RESULTS = ROOT / "results"
CATEGORY_SPLIT = "test_beavertails_30k_test"  # protocol.md, Categories (BeaverTails test only)
META = ("run_id", "config_hash", "condition", "split", "mode")
GATED = ("beavertails_30k_test", "toxicchat_0124_test", "wildguardtest")  # protocol.md, Harness gate
TOLERANCE = 2.0  # percentage points


def flat(d, prefix=""):
    """d with nested dicts turned into dotted keys: {"a": {"b": 1}} -> {"a.b": 1}."""
    out = {}
    for k, v in d.items():
        out |= flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def score_run(manifest, predictions, examples):
    """{run_id, config_hash, condition, split, mode, then the metrics of protocol.md, Metrics, with nested ones as
    dotted keys (confusion.unsafe.invalid, per_category.<key>.f1)} for one run. predictions are its predictions.jsonl
    rows, examples its split's Examples; a row's gold comes from the Example with its example_id. Only on
    CATEGORY_SPLIT: category metrics, and paper.<reading>.f1 / .accuracy for the paper's four BeaverTails readings
    (thinkguard.evaluation.paper), as fractions like every other metric. Raises ValueError if the run did not
    complete, the predictions' example ids are not the split's (missing, extra or repeated), or a row has another
    config_hash than the manifest.
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
        paper = beavertails_scores(
            (e.gold_safety_label, set(e.gold_categories or ()), p.label, p.categories) for e, p, _ in rows
        )
        out |= {
            f"paper.{r}.{k}": v / 100 for r, fa in paper.items() for k, v in zip(("f1", "accuracy"), fa, strict=True)
        }
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


def read_csv(path):
    """path's rows as dicts keyed by its header."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def delta(ours, paper):
    """round(round(100 * ours, 1) - paper, 1): ours (a fraction) minus the paper's number in percentage points, both at
    the paper's one decimal (protocol.md, Harness gate).
    """
    return round(round(100 * ours, 1) - paper, 1)


def gate(scores, targets):
    """results/gate.md: R's decision-only test runs in scores (scores.csv rows) against ThinkGuard's Table 1 F1 and
    AUPRC and Table 3 accuracy in targets (paper_targets.csv rows). The first line is PASS if every benchmark in GATED
    has |delta| <= TOLERANCE (on BeaverTails, F1 and accuracy of the reading paper.choose picks), else FAIL. AUPRC and
    OpenAI Moderation are listed with their delta, not gated. Raises ValueError if a GATED benchmark has no run.
    """
    paper = {(t["benchmark"], t["metric"]): float(t["value"]) for t in targets
             if t["model"] == "ThinkGuard" and t["source"].startswith(("Table 1 ", "Table 3 "))}  # fmt: skip
    runs = {}
    for s in scores:
        if s["condition"] == "R" and s["mode"] == "decision" and s["split"].startswith("test_"):
            run = runs.setdefault(s["split"].removeprefix("test_"), {"run": (s["run_id"], s["config_hash"])})
            run[s["metric"]] = float(s["value"])
    missing = [b for b in GATED if b not in runs]
    if missing:
        raise ValueError(f"no R decision-only run for {missing}")
    passed, rows = True, []
    for b, m in sorted(runs.items()):
        checks = [("F1", m["unsafe_f1"])]
        if b == GATED[0]:  # BeaverTails: the paper's F1 counts categories (reports/paper_protocol.md)
            readings = {r: (100 * m[f"paper.{r}.f1"], 100 * m[f"paper.{r}.accuracy"]) for r in READINGS}
            k = choose(readings, paper[b, "F1"], paper[b, "Accuracy"])
            checks = [(f"F1 (reading {k})", m[f"paper.{k}.f1"]), (f"Accuracy (reading {k})", m[f"paper.{k}.accuracy"])]
        for metric, ours in [*checks, ("AUPRC", m["unsafe_auprc"])]:
            target = paper[b, metric.split()[0]]
            d, gated = delta(ours, target), b in GATED and metric != "AUPRC"
            ok = abs(d) <= TOLERANCE
            passed &= ok or not gated
            verdict = ("yes" if ok else "no") if gated else "-"
            invalid = 100 * (1 - m["coverage"])
            rows.append(f"| {b} | {metric} | {target:.1f} | {100 * ours:.1f} | {d:+.1f} | {verdict} | {invalid:.1f} |")
    bt = [f"| {r} | {f:.1f} | {a:.1f} |" for r, (f, a) in readings.items()]
    ids = [f"| {b} | {m['run'][0]} | {m['run'][1]} |" for b, m in sorted(runs.items())]
    return "\n".join([
        "PASS" if passed else "FAIL", "",
        "R under our harness vs ThinkGuard in the paper (protocol.md, Harness gate). Ours is rounded to one decimal;",
        f"a gated row passes if |Δ| <= {TOLERANCE}. Written by `python -m thinkguard.report gate` from scores.csv.", "",
        "| benchmark | metric | paper | ours | Δ | pass (- = not gated) | invalid % |", "|---|---|---|---|---|---|---|",
        *rows, "", f"BeaverTails under the four readings of reports/paper_protocol.md (reading {k} chosen):", "",
        "| reading | F1 | accuracy |", "|---|---|---|", *bt, "",
        "| benchmark | run_id | config_hash |", "|---|---|---|", *ids, "",
    ])  # fmt: skip


def main(argv=None):
    parser = argparse.ArgumentParser(description="Results tables from stored predictions.")
    tables = parser.add_subparsers(dest="table", required=True)
    scores = tables.add_parser("scores", help="results/scores.csv")
    scores.add_argument("--runs", nargs="+", required=True, help="run ids under THINKGUARD_RUNS")
    tables.add_parser("gate", help="results/gate.md from results/scores.csv")
    args = parser.parse_args(argv)
    if args.table == "gate":
        text = gate(read_csv(RESULTS / "scores.csv"), read_csv(RESULTS / "paper_targets.csv"))
        (RESULTS / "gate.md").write_text(text, encoding="utf-8", newline="\n")
        print(text.splitlines()[0])
        return
    results = []
    for run_id in args.runs:
        manifest = json.loads((RUNS / run_id / "manifest.json").read_text(encoding="utf-8"))
        lines = (RUNS / run_id / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        results.append(score_run(manifest, [json.loads(x) for x in lines], load_split(manifest["config"]["split"])))
    RESULTS.mkdir(exist_ok=True)
    write_scores(results, RESULTS / "scores.csv")


if __name__ == "__main__":
    main()
