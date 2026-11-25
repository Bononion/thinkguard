"""Paired cluster bootstrap of the difference in unsafe F1 between two conditions (protocol.md, Paired differences)."""

import numpy as np

RESAMPLES, SEED = 10_000, 0
CHUNK = 500  # resamples drawn at once: a CHUNK x clusters count matrix stays small


def f1(counts):
    """Unsafe F1 of [..., (tp, fp, fn)] counts, 0 where tp is 0 (metrics.prf)."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def paired_diff(rows, resamples=RESAMPLES, seed=SEED):
    """(diff, low, high, diffs) for the scored rows, given as (cluster, gold, predicted_a, predicted_b): gold "safe" or
    "unsafe", a prediction "safe", "unsafe" or None for an invalid output, which counts as wrong (a false negative on
    gold unsafe, a false positive on gold safe). diff is a's unsafe F1 minus b's; diffs holds one difference per
    resample, each drawing as many clusters as there are, with replacement, from numpy's default_rng(seed), the same
    draw for a and b; low and high are its 2.5th and 97.5th percentiles. Raises ValueError without rows.
    """
    ids = {}
    counts = []  # per cluster: tp, fp, fn of a, then of b
    for cluster, gold, a, b in rows:
        if cluster not in ids:
            ids[cluster] = len(counts)
            counts.append([0] * 6)
        c = counts[ids[cluster]]
        for offset, predicted in ((0, a), (3, b)):
            if gold == "unsafe":
                c[offset + (0 if predicted == "unsafe" else 2)] += 1
            elif predicted != "safe":
                c[offset + 1] += 1
    if not counts:
        raise ValueError("no scored rows")
    counts = np.array(counts, dtype=float)
    k, rng, diffs = len(counts), np.random.default_rng(seed), []
    for start in range(0, resamples, CHUNK):
        drawn = rng.multinomial(k, np.full(k, 1 / k), size=min(CHUNK, resamples - start)) @ counts
        diffs.append(f1(drawn[:, :3]) - f1(drawn[:, 3:]))
    diffs = np.concatenate(diffs)
    total = counts.sum(axis=0)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(f1(total[:3]) - f1(total[3:])), float(low), float(high), diffs
