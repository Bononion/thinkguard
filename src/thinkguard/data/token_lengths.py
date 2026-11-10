"""Token lengths of every split (#51): the evidence for max_length and max_new_tokens (protocol.md, Other choices).

Run from the repo root: python -m thinkguard.data.token_lengths
Writes reports/token_length_report.csv: per split and segment, the row count, nearest-rank percentiles, the max, and
the share of rows longer than each candidate max length. Train and validation are encoded in train_critique mode
(segments prompt, decision, explanation and total, which includes the final <|eot_id|>); the test sets in infer mode
(prompt only, all that evaluation feeds the model). Test lengths are reported, never used to choose (protocol.md).
"""

import csv
import math

from thinkguard.data.manifest import load_split
from thinkguard.data.sources import TEST_SETS
from thinkguard.paths import ROOT
from thinkguard.prompts import encode

REPORT = ROOT / "reports/token_length_report.csv"
PERCENTILES = (50, 90, 95, 99)
CANDIDATES = (1024, 1536, 2048)
HEADER = ("split", "segment", "rows", *(f"p{p}" for p in PERCENTILES), "max", *(f"above_{c}" for c in CANDIDATES))


def report_rows(split, lengths):
    """One CSV row per segment of lengths ({segment: [token counts]}): nearest-rank percentiles (sorted[ceil(p*n) - 1],
    so always an observed integer) and the share above each candidate, to 4 decimals.
    """
    for segment, xs in lengths.items():
        xs = sorted(xs)
        percentiles = (xs[math.ceil(p / 100 * len(xs)) - 1] for p in PERCENTILES)
        above = (f"{sum(x > c for x in xs) / len(xs):.4f}" for c in CANDIDATES)
        yield (split, segment, len(xs), *percentiles, xs[-1], *above)


def lengths(split):
    """{segment: [token counts]} of the split's rows, encoded as the module docstring says."""
    if split.startswith("test_"):
        return {"prompt": [len(encode(e, "infer").input_ids) for e in load_split(split)]}
    encoded = [encode(e, "train_critique") for e in load_split(split)]
    out = {k: [e.spans[k][1] - e.spans[k][0] for e in encoded] for k in ("prompt", "decision", "explanation")}
    return out | {"total": [len(e.input_ids) for e in encoded]}


def main():
    with open(REPORT, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(HEADER)
        for split in ("train", "validation", *(f"test_{s}" for s in TEST_SETS)):
            out.writerows(report_rows(split, lengths(split)))


if __name__ == "__main__":
    main()
