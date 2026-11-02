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

## Metrics

Fixed before any model output is seen. Every metric is computed per test set, from stored predictions, by code that
only reads these rules.

- Scored rows: every row of a test set whose `gold_safety_label` is not null. Every metric below uses the scored rows
  unless it names a smaller denominator.
- An output is invalid when the parser (#52) finds no valid safety label. Invalid outputs stay in every denominator
  and are never mapped to `safe`.

### Safety label

From the label the model generates. The positive class is `unsafe`.

- `unsafe_f1`, `unsafe_precision`, `unsafe_recall`: the primary metrics. An invalid output counts as wrong: as a false
  negative on a gold-unsafe row, as a false positive on a gold-safe row.
- Confusion matrix: gold `safe` / `unsafe` by predicted `safe` / `unsafe` / `invalid`, in counts.
- `prevalence`: gold-unsafe rows / scored rows.
- `coverage`: valid outputs / scored rows. `unsafe_f1_valid`: `unsafe_f1` over the valid outputs only. Both are
  secondary and never replace `unsafe_f1`.

### Unsafe score and AUPRC

- Unsafe score of a row: `log p("unsafe" | x) - log p("safe" | x)`, where `x` is the inference prompt up to the
  decision position (#50) and `log p(label | x)` is the sum of the log-probabilities of the label's tokens in the full
  serialized context (#53). It is never conditioned on gold labels, gold categories or an explanation, and every
  scored row has one, valid output or not.
- `unsafe_auprc`: `sklearn.metrics.average_precision_score(gold == "unsafe", unsafe_score)`, scikit-learn 1.9.1
  (`requirements.txt`). Average precision is a step-wise sum, not an interpolated area under the PR curve, so it is not
  guaranteed equal to the paper's AUPRC.
- The primary F1 is always the generated-label `unsafe_f1`. An F1 from thresholding the unsafe score, if reported, is
  named `score_f1@t`, with `t` fixed on validation, never on a test set.

### Categories (BeaverTails test only)

A row's predicted categories are the parsed categories if its predicted label is `unsafe`, else none; an invalid
output predicts none. Over the 14 BeaverTails categories (`configs/data/sources.yaml`, `dict_keys`):

- `category_macro_f1`: per-category F1 (positive = the category is in the row's set), averaged with equal weight over
  the 14 categories, over all scored rows (3,021). A gold-safe row adds a false positive for every category the model
  predicts on it and nothing otherwise. A category with no gold and no predicted rows scores 0
  (`zero_division=0`).
- `category_exact_match`: share of gold-unsafe rows (1,733) whose predicted category set equals the gold set. Safe
  rows are left out because an empty set on a correctly labeled safe row would count as an exact match.
- Support: the gold rows per category, reported next to each per-category F1.

### Harness gate (#13)

The paper counts an unsafe prediction as correct only if its categories are also correct (Section 4.2), so its F1 is
not `unsafe_f1`. #8 implements the paper's scoring per benchmark from the paper and upstream `ThinkGuard.py` as
`paper_f1`, used only for this gate and for comparisons with the paper's tables, never between our conditions.

R passes on a benchmark if `|paper_f1 - paper F1| <= 2.0` percentage points. The gate covers every benchmark #8 marks
comparable; a benchmark whose paper scoring we cannot match (for example OpenAI Moderation, where upstream
`ThinkGuard.py` encodes mismatches as `-1`) is reported with its difference but not gated.

## Selection rules

Every choice made with data uses the training rows or the validation split (`data/manifests/validation.json`, 750
rows), never a test set.

### Checkpoint selection (L and C)

- Metric: validation `unsafe_f1` (Metrics, Safety label) from decision-only generation, with the same formatter and
  parser as test evaluation (#59). Invalid outputs count as wrong.
- Cadence: every `ceil(steps_per_epoch / 4)` optimizer steps (4 times per epoch) and at the last step, the same for L
  and C. Every scored checkpoint is saved with its validation metrics.
- Best checkpoint: the highest validation `unsafe_f1`; a tie goes to the earlier step.
- The paper saves a checkpoint every 500 steps and does not say how it picks one (Section 4.1): selecting on
  validation is our rule, not the paper's.

### Loss convention (#62)

- Candidates: `token_mean_all` and `segment_mean` with λ = 1, each a pilot-length C run (#61).
- Choice: the candidate whose best checkpoint has the higher validation `unsafe_f1`; if the two are within 0.5
  percentage points, `token_mean_all`. The paper sums the label loss and the critique loss over tokens (Section 3.3,
  Eq. 2-4), which `token_mean_all` is closest to; `segment_mean`, if chosen, is listed under Deviations.
- L and C use the same convention.

### Other choices

- `score_f1@t`, if reported: `t` maximizes the condition's `score_f1` on validation.
- Max length and the truncation rule (#51) are chosen from train and validation token lengths only. Test-set lengths
  are reported (how many test rows the choice truncates) but never used to choose.

### Test-set firewall

- `thinkguard.evaluate` (#9) exits with an error on a split whose name starts with `test_` unless the config file is
  under `configs/frozen/` and committed: `git status --porcelain <config>` prints nothing.
- The run manifest (#33) records the git commit and the config hash, so every test number traces to a committed
  frozen config.
- A frozen config is never edited after its first test run. A change is a new file, and both runs are reported.

### Harness gate fixes (#13)

R runs on the test sets before any of our conditions, so the gate is the one place where a test score can change our
code. A fix after a `FAIL` must name a concrete cause (template, parsing, scoring), applies to every condition, and is
logged in `results/gate.md` with R's numbers before and after. No setting is changed only to move R toward the paper.
Once the gate passes, the evaluation code is tagged `harness-frozen` in git before B, L or C runs on a test set.

### When test numbers are read

- R at the gate (#12, #13); B after the gate (#64); L and C after their checkpoints are frozen (#63).
- The extension's selection rules (the E vs C quality tolerance, the exit-depth and draft-length rule) are added to
  this section and committed before L or C runs on a test set (#17, #18), so no extension rule is written after
  seeing their test numbers.

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
