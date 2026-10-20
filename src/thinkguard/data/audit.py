"""Audit the ThinkGuard critique release: fields, labels, headers, categories, explanations, duplicates.

Run from the repo root: python -m thinkguard.data.audit
Writes reports/data_audit.md and reports/critique_flags.jsonl (one line per flagged row and reason).
Exits 1 if any row has a parse_error or an unknown category.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

CRITIQUE = Path("data/raw/critique/beavertails_critiques_aug.json")
MANIFEST = Path("data/raw_manifest.json")
REPORT = Path("reports/data_audit.md")
FLAGS = Path("reports/critique_flags.jsonl")

FIELDS = ("instruction", "input", "output")
INPUT = re.compile(r"^User: (.*?)\nAgent: (.*)$", re.DOTALL)
LABEL = re.compile(r"^- Safety Assessment: (safe|unsafe)\b", re.MULTILINE)
HEADER = re.compile(r"^- (Safety Assessment|Categories|Explanation):", re.MULTILINE)
CATEGORIES = re.compile(r"^- Categories:(.*)$", re.MULTILINE)
EXPLANATION = re.compile(r"^- Explanation:(.*)", re.MULTILINE | re.DOTALL)
HEADER_SEQUENCES = {("Safety Assessment", "Categories", "Explanation"), ("Safety Assessment", "Explanation")}
TERMINAL = tuple(".!?\"')]")  # an explanation not ending in one of these was cut off
LABEL_ONLY_CHARS = 20  # an explanation shorter than this just restates the label
# A direct verdict on the conversation, e.g. "The conversation is safe", "The response was classified as **unsafe**"
CONTRADICTION = re.compile(
    r"\b(?:conversation|response|interaction|agent'?s? (?:response|reply)|exchange)\s+(?:is|was|has been)\s+"
    r"(?:(?:correctly\s+|appropriately\s+)?(?:assessed|classified|deemed|rated|labeled|considered)\s+(?:as\s+)?)?"
    r"\**(safe|unsafe)\**\b",
    re.IGNORECASE,
)


def load_keys(manifest=MANIFEST):
    """The 14 BeaverTails category keys, as frozen from the 30k_test release."""
    with open(manifest, encoding="utf-8") as f:
        return json.load(f)["beavertails_30k_test"]["dict_keys"]["category"]


def split_pieces(value):
    return [p.strip() for p in value.split(",")]


def regroup(value, keys):
    """Split a Categories value on "," and regroup the pieces left to right by longest match against the keys.

    A key such as "violence,aiding_and_abetting,incitement" contains commas itself. Returns (keys, unknown pieces).
    """
    pieces = split_pieces(value)
    key_pieces = sorted((k.split(",") for k in keys), key=len, reverse=True)
    found, unknown, i = [], [], 0
    while i < len(pieces):
        for kp in key_pieces:
            if pieces[i : i + len(kp)] == kp:
                found.append(",".join(kp))
                i += len(kp)
                break
        else:
            unknown.append(pieces[i])
            i += 1
    return found, unknown


def audit(rows, keys):
    """Run every check. Returns (stats for the report, sorted flags {row, reason, excerpt})."""
    s = Counter()  # counts of rows per finding
    flags = set()  # (row, reason)
    empty_fields, header_seqs, unknown = Counter(), Counter(), Counter()
    instructions, values, pieces, found = set(), set(), set(), set()
    rows_with = defaultdict(list)  # finding -> rows
    by_input = defaultdict(list)
    labels, explanations = [], []
    excerpts = {}  # (row, reason) -> excerpt, where it isn't the end of the explanation

    for i, r in enumerate(rows):
        fields = {k: r.get(k) for k in FIELDS}
        bad = [k for k, v in fields.items() if not isinstance(v, str) or not v.strip()]
        empty_fields.update(bad)
        if bad:
            flags.add((i, "empty_field"))
        inp, out = (v if isinstance(v, str) else "" for v in (fields["input"], fields["output"]))
        if isinstance(fields["instruction"], str):
            instructions.add(fields["instruction"])
        if inp:
            by_input[inp].append(i)
        if not INPUT.match(inp):
            s["input_parse_errors"] += 1
            flags.add((i, "parse_error"))

        m = LABEL.search(out)
        label = m.group(1) if m else None
        labels.append(label)
        if label:
            s[label] += 1
        else:
            s["label_parse_errors"] += 1
            flags.add((i, "parse_error"))

        seq = tuple(HEADER.findall(out))
        header_seqs[seq] += 1
        if seq not in HEADER_SEQUENCES:
            s["bad_header_sequences"] += 1
            flags.add((i, "parse_error"))
        elif label and ("Categories" in seq) != (label == "unsafe"):
            s[f"{label}_misplaced_categories"] += 1
            flags.add((i, "misplaced_categories"))

        m = CATEGORIES.search(out)
        if m:
            value = m.group(1).strip()
            values.add(value)
            pieces.update(split_pieces(value))
            keys_found, unknown_pieces = regroup(value, keys)
            found.update(keys_found)
            if unknown_pieces:
                unknown.update(unknown_pieces)
                flags.add((i, "unknown_category"))

        m = EXPLANATION.search(out)
        text = m.group(1).strip() if m else ""
        explanations.append(text)
        if m is None:
            s["explanation_missing"] += 1  # also a bad header sequence, so already a parse_error
        elif not text:
            s["explanation_empty"] += 1
            flags.add((i, "empty_field"))
        else:
            if not text.endswith(TERMINAL):
                rows_with["truncated"].append(i)
            if len(text) < LABEL_ONLY_CHARS:
                rows_with["label_only"].append(i)
            other = next((c for c in CONTRADICTION.finditer(text) if label and c.group(1).lower() != label), None)
            if other:
                rows_with["contradiction"].append(i)
                excerpts[i, "contradiction"] = text[max(0, other.start() - 100) : other.end() + 100]

    dup_groups = [g for g in by_input.values() if len(g) > 1]
    conflicts = [(a, b) for g in dup_groups for a, b in combinations(g, 2) if labels[a] != labels[b]]
    rows_with["conflicting_duplicate"] = sorted({i for pair in conflicts for i in pair})
    for reason, idx in rows_with.items():
        flags.update((i, reason) for i in idx)

    stats = {
        "rows": len(rows),
        "counts": s,
        "empty_fields": empty_fields,
        "instruction_lengths": sorted(len(t) for t in instructions),
        "header_sequences": header_seqs,
        "categories": {"pieces": len(pieces), "values": len(values), "keys": sorted(found), "unknown": unknown},
        "rows_with": rows_with,
        "duplicate_inputs": len(dup_groups),
        "duplicate_rows": sum(len(g) for g in dup_groups),
        "conflicts": conflicts,
    }
    flag_rows = [
        {"row": i, "reason": reason, "excerpt": excerpts.get((i, reason), explanations[i][-200:])}
        for i, reason in sorted(flags)
    ]
    return stats, flag_rows


def render(stats, keys, n_flags):
    """The "## Critique release" section of reports/data_audit.md."""
    s, cat, rw = stats["counts"], stats["categories"], stats["rows_with"]

    def rows_list(name):
        idx = rw[name]
        return f"{len(idx)}" + (f" (rows {', '.join(map(str, idx))})" if idx else "")

    seqs = "; ".join(f"{' > '.join(seq) or '(none)'}: {n}" for seq, n in sorted(stats["header_sequences"].items()))
    lengths = ", ".join(map(str, stats["instruction_lengths"]))
    unknown = ", ".join(f"`{p}` x{n}" for p, n in sorted(cat["unknown"].items())) or "none"
    conflicts = ", ".join(f"rows {a} and {b}" for a, b in stats["conflicts"])
    lines = [
        "## Critique release",
        "",
        f"Source: `{CRITIQUE.as_posix()}`, {stats['rows']} rows. Flagged rows: `{FLAGS.as_posix()}` ({n_flags} lines).",
        "",
        "Label source: per the ThinkGuard paper, the critics were given the gold BeaverTails label.",
        "",
        "- Fields: rows with an empty or non-string field: "
        + ", ".join(f"{k} {stats['empty_fields'][k]}" for k in FIELDS),
        f"- Instruction: {len(stats['instruction_lengths'])} distinct string(s), length(s) {lengths} chars",
        f"- Input `User: ...\\nAgent: ...`: {s['input_parse_errors']} rows fail to parse",
        (
            f"- Label `- Safety Assessment: safe|unsafe`: unsafe {s['unsafe']}, safe {s['safe']},"
            f" {s['label_parse_errors']} rows fail to parse"
        ),
        f"- Header sequences: {seqs}; {s['bad_header_sequences']} rows with any other sequence",
        (
            f"- Misplaced Categories: {s['safe_misplaced_categories']} safe rows with a Categories line,"
            f" {s['unsafe_misplaced_categories']} unsafe rows without one"
        ),
        (
            f"- Categories: {cat['pieces']} distinct comma pieces, {cat['values']} distinct Categories values;"
            f" pieces regroup into {len(cat['keys'])} of the {len(keys)} keys; unknown pieces: {unknown}"
        ),
        (
            f"- Explanation: {s['explanation_missing']} missing, {s['explanation_empty']} empty,"
            f" {rows_list('label_only')} label-only (under {LABEL_ONLY_CHARS} chars)"
        ),
        f"- Truncated explanations (no terminal `{' '.join(TERMINAL)}`): {rows_list('truncated')}",
        f"- Contradictions (explanation gives a direct verdict for the other label): {rows_list('contradiction')}",
        f"- Duplicates: {stats['duplicate_inputs']} inputs appear more than once ({stats['duplicate_rows']} rows);"
        f" {len(stats['conflicts'])} pair(s) with different labels" + (f": {conflicts}" if conflicts else ""),
    ]
    return "\n".join(lines) + "\n"


def write(stats, flags, keys, report=REPORT, flags_path=FLAGS):
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Data audit\n\n" + render(stats, keys, len(flags)))
    with open(flags_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in flags)


def main():
    with open(CRITIQUE, encoding="utf-8") as f:
        rows = json.load(f)
    keys = load_keys()
    stats, flags = audit(rows, keys)
    write(stats, flags, keys)
    print(render(stats, keys, len(flags)))
    parse_errors = sum(r["reason"] == "parse_error" for r in flags)
    unknown = sum(r["reason"] == "unknown_category" for r in flags)
    if parse_errors or unknown:
        sys.exit(f"FAIL: {parse_errors} rows with a parse_error, {unknown} rows with an unknown category")


if __name__ == "__main__":
    main()
