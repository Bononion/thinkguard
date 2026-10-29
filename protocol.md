# Protocol

## Training data filtering

Applies to the critique rows in `data/interim/critique.jsonl`. `python -m thinkguard.data.filter_train` applies these
rules and writes every row to `data/interim/critique_filtered.jsonl` with its `exclusion_reason` set, and each conflict
to `reports/label_conflicts.jsonl`.

- Duplicates are critique rows with the same `pair_hash`. Their targets are `teacher_safety_label` plus
  `teacher_categories`; the critique text is ignored (two critiques of the same label are both valid).
- Same targets: keep the row with the lowest `raw_record_id`; every other row of the group gets `exclusion_reason`
  `"duplicate"`. No `duplicate_of` field is stored: an excluded duplicate's kept row is the row with the same
  `pair_hash` and the lowest `raw_record_id`.
- Different targets: exclude every row of the group with `exclusion_reason` `"duplicate_conflict"` (no basis to pick
  one).
- Teacher vs human: a row whose `gold_safety_label` is not null and whose gold label or gold categories differ from its
  teacher label or categories gets `exclusion_reason` `"teacher_human_conflict"`. Rows with a null gold label are
  kept: we train on the teacher label.
- One reason per row: the rules are checked in the order above and the first match wins.
