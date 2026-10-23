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
