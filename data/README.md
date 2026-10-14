# Data

## BeaverTails test

- The paper names no BeaverTails test split.
- The 15,000 training rows are 100% from `30k_train` (`scripts/check_source_split.py`).
- Test = `30k_test` (3,021 rows) minus 7 rows that overlap with training = 3,014 rows; unsafe count = 1,729.
- The authors' ThinkGuard.py loads 30k_test, consistent with this choice.

## BeaverTails validation

- Source: `30k_train` (27,186 rows, 27,117 unique keys). A key is (prompt, response) with whitespace collapsed; duplicate rows with agreeing labels count once.
- Excluded keys: all training keys (14,982), test keys (5 also appear in `30k_train`), and keys whose rows have conflicting labels (3, all of them training keys).
- Pool = 12,130 keys, sorted by key; 3,000 sampled with `random.Random(SEED).sample`, `SEED = 42` (`scripts/prep_eval.py`).
- Val = 3,000 rows; unsafe count = 1,717.

## ToxicChat test

- Source: `lmsys/toxic-chat`, config `toxicchat0124`, split `test` (HF revision `29df8e4dba60e1f4af4b4075c0705c5b313548a8`).
- The config is an assumption: the paper and the authors' code don't name one.
- Label: `unsafe` if `toxicity == 1`, else `safe`.
- 5,083 rows; unsafe count = 362.
- Response = `model_output`, kept for a later eval-mode check.

## OpenAI Moderation test

- Source: `mmathys/openai-moderation-api-evaluation`, config `default` (the only one), split `train` (HF revision `84e5cf3bcd6acb3dfc70b6760451645872218a3e`).
- Label: `unsafe` if any of S, H, V, HR, SH, S3, H2, V2 is 1, else `safe`.
- Response: none (`null`); the set is prompt-only.
- 925 rows have at least one missing flag; missing flags are treated as 0, matching the authors' ThinkGuard.py and the benchmark's standard 522 unsafe.
- 1,680 rows; unsafe count = 522.