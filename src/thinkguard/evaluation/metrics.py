"""Metrics of one test set's stored predictions, by protocol.md, Metrics. Every value is a fraction in [0, 1]."""

from sklearn.metrics import average_precision_score

LABELS = ("safe", "unsafe")


def prf(tp, fp, fn):
    """(precision, recall, f1), each 0 when its denominator is 0 (zero_division=0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def safety_metrics(rows):
    """protocol.md, Safety label and Unsafe score and AUPRC, over the scored rows (gold not None).

    rows are (gold, predicted, unsafe_score): gold "safe", "unsafe" or None (not scored); predicted "safe", "unsafe"
    or None for an invalid output; unsafe_score a float. An invalid output is wrong: a false negative on gold unsafe, a
    false positive on gold safe. A ratio whose denominator is 0 is 0. Returns {scored, confusion
    ({gold: {"safe", "unsafe", "invalid": count}}), prevalence, coverage, unsafe_precision, unsafe_recall, unsafe_f1,
    unsafe_f1_valid (over valid outputs only), unsafe_auprc}. Raises ValueError if no row is scored.
    """
    rows = [r for r in rows if r[0] is not None]
    if not rows:
        raise ValueError("no scored rows")
    confusion = {gold: dict.fromkeys((*LABELS, "invalid"), 0) for gold in LABELS}
    for gold, predicted, _ in rows:
        confusion[gold][predicted or "invalid"] += 1
    u, s = confusion["unsafe"], confusion["safe"]
    precision, recall, f1 = prf(u["unsafe"], s["unsafe"] + s["invalid"], u["safe"] + u["invalid"])
    n = len(rows)
    return {
        "scored": n,
        "confusion": confusion,
        "prevalence": sum(u.values()) / n,
        "coverage": (n - u["invalid"] - s["invalid"]) / n,
        "unsafe_precision": precision,
        "unsafe_recall": recall,
        "unsafe_f1": f1,
        "unsafe_f1_valid": prf(u["unsafe"], s["unsafe"], u["safe"])[2],
        "unsafe_auprc": float(average_precision_score([g == "unsafe" for g, _, _ in rows], [x for _, _, x in rows])),
    }
