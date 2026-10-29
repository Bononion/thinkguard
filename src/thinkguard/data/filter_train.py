"""Filter the critique rows for training (exact duplicates, label conflicts, test overlap) by the rules in protocol.md.

Run from the repo root, after normalize: python -m thinkguard.data.filter_train
Writes data/interim/critique_filtered.jsonl (every row in raw order, exclusion_reason set on the excluded ones) and
reports/label_conflicts.jsonl (one line per conflict), and prints the counts.
No timestamps, so a rerun is byte-identical.
"""

import json
from collections import Counter, defaultdict
from dataclasses import replace

from thinkguard.data import audit
from thinkguard.data.normalize import BENCHMARKS, INTERIM, counts, write
from thinkguard.data.schema import read_examples

CRITIQUE = INTERIM / "critique.jsonl"
TEST_KEY = {"prompt_response": "pair_hash", "prompt_only": "prompt_hash"}  # a test row's key, by its task_type


def test_key(t):
    """A test row's key as (field, hash): its pair_hash or prompt_hash, by its own task_type."""
    field = TEST_KEY[t.task_type]
    return field, getattr(t, field)


def targets(e):
    return e.teacher_safety_label, e.teacher_categories


def conflict(kind, rows, gold=False):
    """One reports/label_conflicts.jsonl line: the rows' teacher labels, and their gold labels if gold."""
    fields = ("teacher_safety_label", "teacher_categories") + (("gold_safety_label", "gold_categories") if gold else ())
    return {
        "type": kind,
        "raw_record_ids": [e.raw_record_id for e in rows],
        "rows": [{"raw_record_id": e.raw_record_id, **{f: getattr(e, f) for f in fields}} for e in rows],
    }


def filter_rows(examples, tests):
    """(examples with exclusion_reason set, conflict lines), by the protocol.md rules. tests maps each test set, in
    normalize.BENCHMARKS order, to its rows.
    """
    groups = defaultdict(list)
    for e in examples:
        groups[e.pair_hash].append(e)
    reason, conflicts = {}, []
    for group in groups.values():
        if len(group) == 1:
            continue
        group.sort(key=lambda e: e.raw_record_id)
        if len({targets(e) for e in group}) > 1:
            conflicts.append(conflict("teacher_teacher", group))
            reason.update((e.raw_record_id, "duplicate_conflict") for e in group)
            continue
        for e in group[1:]:
            reason[e.raw_record_id] = "duplicate"
    for e in examples:
        if e.raw_record_id in reason or e.gold_safety_label is None:
            continue
        if (e.gold_safety_label, e.gold_categories) != targets(e):
            conflicts.append(conflict("teacher_human", [e], gold=True))
            reason[e.raw_record_id] = "teacher_human_conflict"
    test_keys = {name: {test_key(t) for t in rows} for name, rows in tests.items()}
    for e in examples:
        if e.raw_record_id in reason:
            continue
        keys = {(f, getattr(e, f)) for f in TEST_KEY.values()}
        overlap = next((name for name, k in test_keys.items() if keys & k), None)
        if overlap:
            reason[e.raw_record_id] = f"test_overlap_{overlap}"
    filtered =[replace(e, exclusion_reason=reason.get(e.raw_record_id)) for e in examples]
    return filtered, conflicts


def main():
    examples = read_examples(CRITIQUE)
    tests = {name: read_examples(INTERIM / f"{name}.jsonl") for name in BENCHMARKS}
    filtered, conflicts = filter_rows(examples, tests)
    write(filtered, audit.TRAIN_FILTERED)
    audit.LABEL_CONFLICTS.parent.mkdir(parents=True, exist_ok=True)
    with open(audit.LABEL_CONFLICTS, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in conflicts)

    reasons = Counter(e.exclusion_reason for e in filtered)
    types = Counter(c["type"] for c in conflicts)
    group_sizes = [n for n in Counter(e.pair_hash for e in examples).values() if n > 1]
    print(
        f"critique: {len(filtered)} rows, {reasons[None]} kept;"
        f" {len(group_sizes)} duplicate groups ({sum(group_sizes)} rows);"
        f" conflicts {counts({t: types[t] for t in audit.CONFLICT_TYPES})};"
        f" exclusions {counts({r: reasons[r] for r in audit.EXCLUSION_REASONS})}"
    )


if __name__ == "__main__":
    main()
