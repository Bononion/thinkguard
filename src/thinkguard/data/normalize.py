"""Normalize every source into Example rows, one JSON line each in data/interim/<source>.jsonl.

Run from the repo root: python -m thinkguard.data.normalize
Rows that fail to parse are dropped and counted by parse_status. Benchmark gold labels follow each source's unsafe
rule in configs/data/sources.yaml. Critique rows take their teacher label from the critique and their gold label
from the beavertails_30k_train annotations of the same pair; neither is ever filled from the other.
Output keeps raw row order and has no timestamps, so a rerun is byte-identical.
Exits 1 if a benchmark's gold unsafe count differs from its unsafe_rows in sources.yaml.
"""

import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from thinkguard.data import audit
from thinkguard.data.keys import pair_hash, prompt_hash
from thinkguard.data.schema import Example
from thinkguard.data.sources import is_unsafe, load, load_sources, unsafe_fields

GOLD_SPLIT = "beavertails_30k_train"  # the critique release's source split (#42)
INTERIM = Path("data/interim")


def missing_text(row, fields):
    """The first field that is missing, not a string, or blank, as a parse_status; None if all have text."""
    return next((f"{f}_missing" for f in fields if not isinstance(row.get(f), str) or not row[f].strip()), None)


def gold_label(row, rule):
    """Gold "unsafe" / "safe" by a source's unsafe rule, or None if every field of the rule is missing or null."""
    if all(row.get(f) is None for f in unsafe_fields(rule)):
        return None
    return "unsafe" if is_unsafe(row, rule) else "safe"


def true_keys(flags):
    """The keys of a BeaverTails category dict that are true, sorted."""
    return tuple(sorted(k for k, v in flags.items() if v))


def benchmark(prompt, response, gold, categories, status):
    """The fields of a benchmark row: gold only, no teacher. A row with no gold label is "missing" and has no
    categories either (None; an empty tuple is a label with no categories).
    """
    return {
        "user_prompt": prompt,
        "assistant_response": response,
        "task_type": "prompt_only" if response is None else "prompt_response",
        "gold_safety_label": gold,
        "gold_categories": None if gold is None else categories,
        "teacher_safety_label": None,
        "teacher_categories": None,
        "critique": None,
        "annotation_status": "missing" if gold is None else status,
    }


def beavertails(row, rule):
    if reason := missing_text(row, ("prompt", "response")):
        return reason
    return benchmark(row["prompt"], row["response"], gold_label(row, rule), true_keys(row["category"]), "human")


def toxicchat(row, rule):
    if reason := missing_text(row, ("user_input",)):
        return reason
    status = "human" if row["human_annotation"] else "auto"  # datasets reads the CSV's True/False as bool
    return benchmark(row["user_input"], None, gold_label(row, rule), (), status)


def openai_moderation(row, rule):
    if reason := missing_text(row, ("prompt",)):
        return reason
    flagged = tuple(sorted(f for f in unsafe_fields(rule) if row.get(f) == rule["equals"]))
    return benchmark(row["prompt"], None, gold_label(row, rule), flagged, "human")


def wildguardtest(row, rule):
    if reason := missing_text(row, ("prompt", "response")):
        return reason
    return benchmark(row["prompt"], row["response"], gold_label(row, rule), (), "human")


def gold_annotations(rows, rule):
    """pair_hash -> the distinct (gold label, categories) among its BeaverTails annotations."""
    idx = defaultdict(set)
    for r in rows:
        idx[pair_hash(r["prompt"], r["response"])].add((gold_label(r, rule), true_keys(r["category"])))
    return idx


def critique(row, keys, gold):
    """Teacher fields parsed from the critique; gold from gold_annotations of the same pair, if they all agree."""
    m = audit.INPUT.match(row["input"])
    if m is None:
        return "input_unparsed"
    label = audit.LABEL.search(row["output"])
    if label is None:
        return "label_unparsed"
    cats = audit.CATEGORIES.search(row["output"])
    found, unknown = audit.regroup(cats.group(1).strip(), keys) if cats else ([], [])
    if unknown:
        return "unknown_category"
    prompt, response = m.groups()
    annotations = gold.get(pair_hash(prompt, response), set())
    if len(annotations) == 1:
        [(gold_safety, gold_categories)] = annotations
        status = "human"
    else:
        gold_safety, gold_categories = None, None
        status = "conflicting" if annotations else "missing"
    return {
        "user_prompt": prompt,
        "assistant_response": response,
        "task_type": "prompt_response",
        "gold_safety_label": gold_safety,
        "gold_categories": gold_categories,
        "teacher_safety_label": label.group(1),
        "teacher_categories": tuple(sorted(found)),
        "critique": row["output"],
        "annotation_status": status,
    }


BENCHMARKS = {
    "beavertails_30k_test": beavertails,
    "toxicchat_0124_test": toxicchat,
    "openai_moderation": openai_moderation,
    "wildguardtest": wildguardtest,
}
SOURCES = ("critique", *BENCHMARKS)


def normalize(name, source, rows, to_fields):
    """(Examples of the rows that parse, parse_status counts over every raw row)."""
    examples, status = [], Counter()
    for i, row in enumerate(rows):
        fields = to_fields(row)
        if isinstance(fields, str):
            status[fields] += 1
            continue
        status["ok"] += 1
        prompt, response = fields["user_prompt"], fields["assistant_response"]
        examples.append(
            Example(
                example_id=f"{name}:{i:06d}",
                source_dataset=name,
                source_revision=source["revision"],
                source_split=source["split"],
                raw_record_id=i,
                prompt_hash=prompt_hash(prompt),
                pair_hash=pair_hash(prompt, response) if fields["task_type"] == "prompt_response" else None,
                parse_status="ok",
                exclusion_reason=None,
                **fields,
            )
        )
    return examples, status


def write(examples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(e.to_json() + "\n" for e in examples)


def counts(counter):
    return ", ".join(f"{k} {n}" for k, n in sorted(counter.items())) or "none"


def main():
    sources = {s["name"]: s for s in load_sources()}
    gold = gold_annotations(load(GOLD_SPLIT), sources[GOLD_SPLIT]["unsafe"])
    to_fields = {name: partial(fn, rule=sources[name]["unsafe"]) for name, fn in BENCHMARKS.items()}
    to_fields["critique"] = partial(critique, keys=audit.load_keys(), gold=gold)

    mismatches = []
    for name in SOURCES:
        rows = load(name)
        examples, status = normalize(name, sources[name], rows, to_fields[name])
        write(examples, INTERIM / f"{name}.jsonl")
        dropped = Counter({k: n for k, n in status.items() if k != "ok"})
        unsafe = sum(e.gold_safety_label == "unsafe" for e in examples)
        print(
            f"{name}: {len(rows)} raw, {len(examples)} written, {dropped.total()} dropped ({counts(dropped)});"
            f" gold unsafe {unsafe}; annotation_status {counts(Counter(e.annotation_status for e in examples))}"
        )
        if name in BENCHMARKS and unsafe != sources[name]["unsafe_rows"]:
            mismatches.append(f"{name} {unsafe} != {sources[name]['unsafe_rows']}")
    if mismatches:
        sys.exit(f"FAIL: gold unsafe count differs from sources.yaml's unsafe_rows: {'; '.join(mismatches)}")


if __name__ == "__main__":
    main()
