# Data audit

## Critique release

Source: `https://raw.githubusercontent.com/luka-group/ThinkGuard/a7d0c8924a343009e9fd4b31aa84ac845b1ff454/data/beavertails_critiques_aug.json`, 15000 rows. Flagged rows: `reports/critique_flags.jsonl` (8 lines).

Label source: per the ThinkGuard paper, the critics were given the gold BeaverTails label.

- Fields: rows with an empty or non-string field: instruction 0, input 0, output 0
- Instruction: 1 distinct string(s), length(s) 631 chars
- Input `User: ...\nAgent: ...`: 0 rows fail to parse
- Label `- Safety Assessment: safe|unsafe`: unsafe 8526, safe 6474, 0 rows fail to parse
- Header sequences: Safety Assessment > Categories > Explanation: 8526; Safety Assessment > Explanation: 6474; 0 rows with any other sequence
- Misplaced Categories: 0 safe rows with a Categories line, 0 unsafe rows without one
- Categories: 27 distinct comma pieces, 185 distinct Categories values; pieces regroup into 14 of the 14 keys; unknown pieces: none
- Explanation: 0 missing, 0 empty, 0 label-only (under 20 chars)
- Truncated explanations (no terminal `. ! ? " ' ) ]`): 4 (rows 1324, 3161, 3245, 10557)
- Contradictions (explanation gives a direct verdict for the other label): 2 (rows 2561, 14857)
- Duplicates: 18 inputs appear more than once (36 rows); 1 pair(s) with different labels: rows 4774 and 5475

## Training filter

Source: `data/interim/critique_filtered.jsonl`, 15000 rows; 14974 kept. Rules: `protocol.md`, Training data filtering. Conflicts: `reports/label_conflicts.jsonl`.

- Duplicates (same pair_hash): 18 groups (36 rows)
- Conflicts: teacher_teacher 2, teacher_human 0
- Exclusions by reason: duplicate 16, duplicate_conflict 4, teacher_human_conflict 0, test_overlap_beavertails_30k_test 6, test_overlap_toxicchat_0124_test 0, test_overlap_openai_moderation 0, test_overlap_wildguardtest 0
- Teacher labels, before -> after the test-overlap rule: unsafe 8515 -> 8512, safe 6465 -> 6462

Test overlap: critique rows whose pair_hash / prompt_hash matches a test row's, before -> after the test-overlap rule (before: exclusion_reason null or test_overlap_*; after: null). Each test row's key follows its task_type, so a prompt_response set's prompt-only overlap stays. Last column: test rows whose prompt_hash is in the critique rows kept after.

| Test set | Rows | pair_hash overlap | prompt_hash overlap | Test rows with prompt in training |
|---|---|---|---|---|
| `beavertails_30k_test` | 3021 | 6 -> 0 | 4495 -> 4489 | 2552 (469 unseen) |
| `toxicchat_0124_test` | 5083 | n/a | 0 -> 0 | 0 (5083 unseen) |
| `openai_moderation` | 1680 | n/a | 0 -> 0 | 0 (1680 unseen) |
| `wildguardtest` | 1725 | 0 -> 0 | 0 -> 0 | 0 (1725 unseen) |

## Benchmarks

| Benchmark | Gold-label field(s) | Rows | Positives | Missing label cells |
|---|---|---|---|---|
| `toxicchat_0124_test` | `toxicity` | 5083 | 362 | 0 |
| `openai_moderation` | `S`, `H`, `V`, `HR`, `SH`, `S3`, `H2`, `V2` | 1680 | 522 | 4142 |
| `wildguardtest` | `response_harm_label` | 1725 | 284 | 16 |

- ToxicChat uses toxicchat0124 (newer version; the paper names none; final choice in #48).
- OpenAI Moderation is unsafe if any of S, H, V, HR, SH, S3, H2, V2 equals 1. The upstream README gives no binary rule; this follows AI2's safety-eval (used for the WildGuard paper), which gives the same 522 positives.
- OpenAI Moderation: 925 rows lack a label for at least one category. Upstream says a missing label means unknown; we count it as not flagged, as safety-eval does. No row lacks all 8.
- WildGuardTest: gold label is response_harm_label (provisional; our models classify the agent turn, matching safety-eval's wildguardtest_response task). prompt_harm_label is the alternative; final choice in #48.
