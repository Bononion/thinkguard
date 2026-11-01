# Protocol

## Conditions

Checkpoint revisions are pinned in `configs/frozen/source_manifest.json`. Every condition decodes greedily
(`do_sample=False`) with the same prompt contract (#48).

| ID | Checkpoint | Decoding | Role |
|---|---|---|---|
| B | `meta-llama/Llama-Guard-3-8B` | autoregressive | base model |
| L | B + `label_category` LoRA | autoregressive | label-only ablation of C |
| C | B + critique LoRA | autoregressive | our ThinkGuard reproduction |
| R | `Rakancorle1/ThinkGuard` (released) | autoregressive | reference only |

- In code the label-only condition is named `label_category`: it keeps category supervision and drops only the
  critique.
- The early-exit extension's conditions are added once the reproduction is done (#63).

## Research questions

Each question is answered by one comparison, read from one generated artifact. Metrics are defined in #28; paired
differences use the cluster bootstrap (#65).

| # | Question | Comparison | Metric | Artifact |
|---|---|---|---|---|
| 1 | Does training on critiques improve the guard over training on labels alone? | L vs C | unsafe F1 per test set; BeaverTails category macro F1 | `results/table_reproduction.md` |
| 2 | Does LoRA adaptation improve over the base guard? | B vs L, B vs C | unsafe F1, precision, recall, AUPRC per test set | `results/table_reproduction.md` |
| 3 | How close is our reproduction to the released ThinkGuard and the paper? | R vs C (reference only) | unsafe F1 per test set | `results/table_reproduction.md` |

R is a reference: it is trained with full fine-tuning on data and settings we cannot fully check, so question 3 gets
no significance test and no claim beyond "close" or "not close".

## Deviations

- LoRA, not the paper's full fine-tuning, is a project deviation: C reproduces ThinkGuard's method, not its exact
  weights.

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
- Smoke (32 rows) and pilot (1000 rows) subsets are drawn only from the train side of the validation split.
- Both keep train's unsafe ratio: `n_unsafe = round(size * unsafe / rows)`, with `unsafe` and `rows` counted over
  train; the rest are safe.
- The pilot includes the 20 longest train rows by `len(user_prompt) + len(assistant_response) + len(critique)`, ties
  broken by `example_id`; they count toward its unsafe / safe totals.
- The other rows are drawn per label with `random.Random(0).sample` (a new generator for each label) from that label's
  remaining train rows sorted by `example_id`.
- Each test set's manifest lists every row of its interim file, including rows with no gold label; which rows are
  scored is decided in #48.
