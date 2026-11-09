"""Results tables from stored predictions, never from re-generation (#54).

score_run scores one run from its manifest, its predictions.jsonl rows and its split's Examples.
"""

from collections import Counter

from thinkguard.evaluation.metrics import category_metrics, flag_rates, safety_metrics
from thinkguard.evaluation.parse import parse

CATEGORY_SPLIT = "test_beavertails_30k_test"  # protocol.md, Categories (BeaverTails test only)


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
