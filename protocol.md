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
- Test overlap: a row still kept after the rules above that shares a key with a row of a test set gets
  `exclusion_reason` `"test_overlap_<set>"`. The test sets are the four benchmarks in `sources.TEST_SETS`, read from
  `data/interim/<set>.jsonl`. A test row's key follows its `task_type`: `pair_hash` for `prompt_response` rows
  (`beavertails_30k_test`, `wildguardtest`), `prompt_hash` for `prompt_only` rows (`toxicchat_0124_test`,
  `openai_moderation`). A row that matches several sets takes the first set in `sources.TEST_SETS` order.
- Same prompt with a different response is not excluded for `prompt_response` sets: the model labels the response,
  and BeaverTails reuses each prompt with many responses by design, so excluding it would drop about 30% of training
  rows. The count is reported instead (`reports/data_audit.md`, Training filter), so the test rows whose prompt never
  appears in training can be scored separately.
- One reason per row: the rules are checked in the order above and the first match wins.

## Validation split

Applies to the rows of `data/interim/critique_filtered.jsonl` whose `exclusion_reason` is null (the rows kept for
training). `thinkguard.data.split` applies these rules; `python -m thinkguard.data.audit` reports the result in
`reports/data_audit.md`, Validation split.

- Validation is about 5% of the kept training rows, grouped by `prompt_hash`: every row with the same prompt goes to
  the same side, so no prompt appears in both.
- Split: sort the distinct `prompt_hash` values, shuffle them with `random.Random(seed)`, and move whole groups to
  validation in that order until it has at least 5% of the rows. The rest is train.
- Seed rule: try seed 0, 1, 2, ... and use the first seed where every BeaverTails category (`teacher_categories`) has
  at least 5 validation rows; if no seed below 100 passes, the split fails. The seeds tried and why they failed are
  logged in `reports/data_audit.md`.
