"""Find which BeaverTails split the critique release came from, by matching its conversations on pair_hash.

Run from the repo root, after the download and a passing audit: python -m thinkguard.data.match
Report only; no labels change. Writes reports/source_match.md: per-split match counts, the source split, the
critique rows whose pair has conflicting human is_safe annotations there, and the rows not found in it.
Exits 1 unless exactly one split contains every critique row.
"""

import json
from collections import defaultdict
from pathlib import Path

from thinkguard.data import audit, download
from thinkguard.data.keys import pair_hash

SPLITS = ("beavertails_30k_train", "beavertails_30k_test", "beavertails_330k_train", "beavertails_330k_test")
REPORT = Path("reports/source_match.md")


def read_split(name, manifest):
    path = download.target_path(name, manifest[name]["url"])
    reader, _ = download.READERS[path.name.split(".", 1)[1]]  # format = the file's extensions, e.g. jsonl.xz
    return reader(path)


def index(rows):
    """pair_hash -> [is_safe], one item per annotation."""
    idx = defaultdict(list)
    for r in rows:
        idx[pair_hash(r["prompt"], r["response"])].append(r["is_safe"])
    return idx


def critique_hashes(rows):
    """pair_hash per critique row, split with audit.py's input parser."""
    return [pair_hash(*audit.INPUT.match(r["input"]).groups()) for r in rows]


def conflicting(annotations):
    return len(set(annotations)) > 1


def match(hashes, indexes):
    """All the numbers of the report. source is None unless exactly one split contains every critique row."""
    splits = {
        name: {
            "rows": sum(map(len, idx.values())),
            "pairs": len(idx),
            "multi": sum(len(a) > 1 for a in idx.values()),
            "conflicting": sum(map(conflicting, idx.values())),
            "matched": sum(h in idx for h in hashes),
        }
        for name, idx in indexes.items()
    }
    full = [name for name, s in splits.items() if s["matched"] == len(hashes)]
    best = full[0] if len(full) == 1 else max(splits, key=lambda n: splits[n]["matched"])
    idx = indexes[best]
    return {
        "splits": splits,
        "source": best if len(full) == 1 else None,
        "full": full,
        "best": best,
        "conflict_rows": [i for i, h in enumerate(hashes) if h in idx and conflicting(idx[h])],
        "unmatched": [i for i, h in enumerate(hashes) if h not in idx],
    }


def pct(n, total):
    return f"{100 * n / total:.2f}%" if total else "n/a"


def rows_list(rows):
    return f"{len(rows)}" + (f" (rows {', '.join(map(str, rows))})" if rows else "")


def render(stats, n_rows):
    """reports/source_match.md."""
    lines = [
        "# Source match",
        "",
        (
            "Critique rows are matched to BeaverTails on `pair_hash` (NFC, whitespace-collapsed prompt and response;"
            " `thinkguard/data/keys.py`). Report only: no labels are changed."
        ),
        "",
        (
            "| Split | Rows | Distinct pairs | Pairs with >1 annotation | Pairs with conflicting is_safe"
            f" | {n_rows} critique rows matched | Match |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in stats["splits"].items():
        lines.append(
            f"| {name} | {s['rows']} | {s['pairs']} | {s['multi']} | {s['conflicting']} | {s['matched']}"
            f" | {pct(s['matched'], n_rows)} |"
        )
    source, best = stats["source"], stats["best"]
    lines.append("")
    if source is None:
        lines.append(f"No single source split: splits matching every row: {', '.join(stats['full']) or 'none'}.")
    else:
        others = [f"{n} ({s['matched']} rows)" for n, s in stats["splits"].items() if n != source and s["matched"]]
        lines += [
            (
                f"Source split: **{source}** (matches all {n_rows} rows)."
                f" Other splits containing critique rows: {', '.join(others) or 'none'}."
            ),
            "",
            (
                f"Critique rows whose pair has conflicting is_safe annotations in {source}:"
                f" {rows_list(stats['conflict_rows'])}"
            ),
        ]
    lines += ["", f"Critique rows not found in {best}: {rows_list(stats['unmatched'])}"]
    return "\n".join(lines) + "\n"


def main():
    with open(audit.CRITIQUE, encoding="utf-8") as f:
        crit = json.load(f)
    with open(download.MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    hashes = critique_hashes(crit)
    stats = match(hashes, {name: index(read_split(name, manifest)) for name in SPLITS})

    text = render(stats, len(crit))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    if stats["source"] is None:
        raise SystemExit(f"FAIL: expected exactly one split matching all {len(crit)} rows, got {stats['full']}")


if __name__ == "__main__":
    main()
