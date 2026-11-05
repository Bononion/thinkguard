"""Audit the ThinkGuard critique release: fields, labels, headers, categories, explanations.

Run from the repo root: uv run python -m thinkguard.data.audit
Run order: normalize, filter_train, audit (each as uv run python -m thinkguard.data.<name>).
Writes reports/data_audit.md and reports/critique_flags.jsonl (one line per flagged row and reason). The critique and
benchmark rows come from thinkguard.data.sources.load; the Benchmarks section counts them with the unsafe rules in
configs/data/sources.yaml. The Training filter section counts data/interim/critique_filtered.jsonl and
reports/label_conflicts.jsonl (written by filter_train), and the test sets' data/interim/<name>.jsonl (by normalize).
The Validation split section splits the kept rows of critique_filtered.jsonl with thinkguard.data.split.
Exits 1 if any row has a parse_error or an unknown category.
"""

import json
import re
import sys
from collections import Counter, defaultdict

from thinkguard.data.critique_format import (
    CATEGORIES,
    CONTRADICTION,
    INPUT,
    LABEL,
    load_keys,
    regroup,
    split_pieces,
)
from thinkguard.data.filter_train import CONFLICT_TYPES, EXCLUSION_REASONS, LABEL_CONFLICTS, TRAIN_FILTERED
from thinkguard.data.schema import read_examples
from thinkguard.data.sources import TEST_SETS, is_unsafe, load, load_sources, unsafe_fields, url
from thinkguard.data.split import MIN_PER_CATEGORY, choose_seed, group_split
from thinkguard.paths import ROOT, rel

REPORT = ROOT / "reports/data_audit.md"
FLAGS = ROOT / "reports/critique_flags.jsonl"

BENCHMARK_TABLE = ("toxicchat_0124_test", "openai_moderation", "wildguardtest")  # BeaverTails: data/README.md

FIELDS = ("instruction", "input", "output")
HEADER = re.compile(r"^- (Safety Assessment|Categories|Explanation):", re.MULTILINE)
EXPLANATION = re.compile(r"^- Explanation:(.*)", re.MULTILINE | re.DOTALL)
HEADER_SEQUENCES = {("Safety Assessment", "Categories", "Explanation"), ("Safety Assessment", "Explanation")}
TERMINAL = tuple(".!?\"')]")  # an explanation not ending in one of these was cut off
LABEL_ONLY_CHARS = 20  # an explanation shorter than this just restates the label


def audit(rows, keys):
    """Run every check. Returns (stats for the report, sorted flags {row, reason, excerpt})."""
    s = Counter()  # counts of rows per finding
    flags = set()  # (row, reason)
    empty_fields, header_seqs, unknown = Counter(), Counter(), Counter()
    instructions, values, pieces, found = set(), set(), set(), set()
    rows_with = defaultdict(list)  # finding -> rows
    explanations = []
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
        if not INPUT.match(inp):
            s["input_parse_errors"] += 1
            flags.add((i, "parse_error"))

        m = LABEL.search(out)
        label = m.group(1) if m else None
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
    }
    flag_rows = [
        {"row": i, "reason": reason, "excerpt": excerpts.get((i, reason), explanations[i][-200:])}
        for i, reason in sorted(flags)
    ]
    return stats, flag_rows


def render(stats, keys, n_flags, source):
    """The "## Critique release" section of reports/data_audit.md; source is the critique release's url."""
    s, cat, rw = stats["counts"], stats["categories"], stats["rows_with"]

    def rows_list(name):
        idx = rw[name]
        return f"{len(idx)}" + (f" (rows {', '.join(map(str, idx))})" if idx else "")

    seqs = "; ".join(f"{' > '.join(seq) or '(none)'}: {n}" for seq, n in sorted(stats["header_sequences"].items()))
    lengths = ", ".join(map(str, stats["instruction_lengths"]))
    unknown = ", ".join(f"`{p}` x{n}" for p, n in sorted(cat["unknown"].items())) or "none"
    lines = [
        "## Critique release",
        "",
        f"Source: `{source}`, {stats['rows']} rows. Flagged rows: `{rel(FLAGS)}` ({n_flags} lines).",
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
    ]
    return "\n".join(lines) + "\n"


def render_train_filter(rows, conflicts, tests):
    """The "## Training filter" section: rows are the Examples of critique_filtered.jsonl, conflicts the lines of
    label_conflicts.jsonl, tests each test set's Examples from data/interim/<name>.jsonl. Before -> after the
    test-overlap rule: the rows whose exclusion_reason is null or test_overlap_* -> null.
    """
    reasons = Counter(r.exclusion_reason for r in rows)
    types = Counter(c["type"] for c in conflicts)
    groups = [n for n in Counter(r.pair_hash for r in rows).values() if n > 1]
    before = [r for r in rows if r.exclusion_reason is None or r.exclusion_reason.startswith("test_overlap_")]
    after = [r for r in rows if r.exclusion_reason is None]
    label_before, label_after = (Counter(r.teacher_safety_label for r in part) for part in (before, after))
    trained_prompts = {r.prompt_hash for r in after}

    def overlap(key, test):
        values = {getattr(t, key) for t in test}
        return " -> ".join(str(sum(getattr(r, key) in values for r in part)) for part in (before, after))

    table = []
    for name, test in tests.items():
        pair = "n/a" if all(t.task_type == "prompt_only" for t in test) else overlap("pair_hash", test)
        seen = sum(t.prompt_hash in trained_prompts for t in test)
        table.append(
            f"| `{name}` | {len(test)} | {pair} | {overlap('prompt_hash', test)} | {seen} ({len(test) - seen} unseen) |"
        )
    lines = [
        "## Training filter",
        "",
        (
            f"Source: `{rel(TRAIN_FILTERED)}`, {len(rows)} rows; {reasons[None]} kept."
            f" Rules: `protocol.md`, Training data filtering. Conflicts: `{rel(LABEL_CONFLICTS)}`."
        ),
        "",
        f"- Duplicates (same pair_hash): {len(groups)} groups ({sum(groups)} rows)",
        f"- Conflicts: {', '.join(f'{t} {types[t]}' for t in CONFLICT_TYPES)}",
        f"- Exclusions by reason: {', '.join(f'{r} {reasons[r]}' for r in EXCLUSION_REASONS)}",
        (
            f"- Teacher labels, before -> after the test-overlap rule: unsafe {label_before['unsafe']} ->"
            f" {label_after['unsafe']}, safe {label_before['safe']} -> {label_after['safe']}"
        ),
        "",
        (
            "Test overlap: critique rows whose pair_hash / prompt_hash matches a test row's, before -> after the"
            " test-overlap rule (before: exclusion_reason null or test_overlap_*; after: null). Each test row's key"
            " follows its task_type, so a prompt_response set's prompt-only overlap stays. Last column: test rows whose"
            " prompt_hash is in the critique rows kept after."
        ),
        "",
        "| Test set | Rows | pair_hash overlap | prompt_hash overlap | Test rows with prompt in training |",
        "|---|---|---|---|---|",
        *table,
    ]
    return "\n".join(lines) + "\n"


def render_split(examples, keys):
    """The "## Validation split" section: examples are the kept rows of critique_filtered.jsonl, split with the seed
    choose_seed picks; keys order the category table.
    """
    seed, failures = choose_seed(examples)
    sides = dict(zip(("Train", "Validation"), group_split(examples, seed=seed), strict=True))
    rejected = "; ".join(f"seed {s} ({', '.join(f'{c} {n}' for c, n in low.items())})" for s, low in failures.items())
    lines = [
        "## Validation split",
        "",
        f"Source: the {len(examples)} kept rows of `{rel(TRAIN_FILTERED)}`. Rules: `protocol.md`, Validation split.",
        "",
        f"- Seed: {seed}. Rejected (a category under {MIN_PER_CATEGORY} validation rows): {rejected or 'none'}",
    ]
    for side, rows in sides.items():
        labels = Counter(e.teacher_safety_label for e in rows)
        lines.append(
            f"- {side}: {len(rows)} rows, {len({e.prompt_hash for e in rows})} prompt groups;"
            f" unsafe {labels['unsafe']}, safe {labels['safe']}"
        )
    cats = {side: Counter(c for e in rows for c in e.teacher_categories) for side, rows in sides.items()}
    lines += ["", "| Category | Train | Validation |", "|---|---|---|"]
    lines += [f"| `{k}` | {cats['Train'][k]} | {cats['Validation'][k]} |" for k in keys]
    return "\n".join(lines) + "\n"


def render_benchmarks(rows, sources):
    """The "## Benchmarks" section, counted from each benchmark's rows. Missing label cells: null or missing
    gold-label values, summed over the gold-label fields.
    """
    lines = [
        "## Benchmarks",
        "",
        "| Benchmark | Gold-label field(s) | Rows | Positives | Missing label cells |",
        "|---|---|---|---|---|",
    ]
    for name in BENCHMARK_TABLE:
        rule = sources[name]["unsafe"]
        fields = unsafe_fields(rule)
        positives = sum(is_unsafe(r, rule) for r in rows[name])
        missing = sum(r.get(f) is None for r in rows[name] for f in fields)
        gold = ", ".join(f"`{f}`" for f in fields)
        lines.append(f"| `{name}` | {gold} | {len(rows[name])} | {positives} | {missing} |")
    notes = benchmark_notes(rows["openai_moderation"], sources["openai_moderation"]["unsafe"])
    return "\n".join([*lines, "", *notes]) + "\n"


def benchmark_notes(moderation, rule):
    """The notes under the Benchmarks table. OpenAI Moderation counts come from its rows; 522 is safety-eval's count,
    which sources.load already enforces as unsafe_rows.
    """
    fields = unsafe_fields(rule)
    missing = [sum(r.get(f) is None for f in fields) for r in moderation]
    some, every = sum(n > 0 for n in missing), sum(n == len(fields) for n in missing)
    lack_all = f"{every} rows lack all {len(fields)}." if every else f"No row lacks all {len(fields)}."
    return (
        "- ToxicChat uses toxicchat0124 (newer version; the paper names none; final choice in #48).",
        f"- OpenAI Moderation is unsafe if any of {', '.join(fields)} equals {rule['equals']}. The upstream README"
        " gives no binary rule; this follows AI2's safety-eval (used for the WildGuard paper), which gives the same"
        " 522 positives.",
        f"- OpenAI Moderation: {some} rows lack a label for at least one category. Upstream says a missing label means"
        f" unknown; we count it as not flagged, as safety-eval does. {lack_all}",
        "- WildGuardTest: gold label is response_harm_label (provisional; our models classify the agent turn, matching"
        " safety-eval's wildguardtest_response task). prompt_harm_label is the alternative; final choice in #48.",
    )


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write(text, flags, report=REPORT, flags_path=FLAGS):
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(flags_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in flags)


def main():
    sources = {s["name"]: s for s in load_sources()}
    keys = load_keys()
    stats, flags = audit(load("critique"), keys)
    benchmarks = render_benchmarks({name: load(name) for name in BENCHMARK_TABLE}, sources)
    filtered = read_examples(TRAIN_FILTERED)
    tests = {name: read_examples(TRAIN_FILTERED.parent / f"{name}.jsonl") for name in TEST_SETS}
    train_filter = render_train_filter(filtered, read_jsonl(LABEL_CONFLICTS), tests)
    kept = [e for e in filtered if e.exclusion_reason is None]
    critique = render(stats, keys, len(flags), url(sources["critique"]))
    text = "# Data audit\n\n" + "\n".join((critique, train_filter, render_split(kept, keys), benchmarks))
    write(text, flags)
    print(text)
    parse_errors = sum(r["reason"] == "parse_error" for r in flags)
    unknown = sum(r["reason"] == "unknown_category" for r in flags)
    if parse_errors or unknown:
        sys.exit(f"FAIL: {parse_errors} rows with a parse_error, {unknown} rows with an unknown category")


if __name__ == "__main__":
    main()
