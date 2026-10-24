# Data audit

## Critique release

Source: `data/raw/critique/beavertails_critiques_aug.json`, 15000 rows. Flagged rows: `reports/critique_flags.jsonl` (8 lines).

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
