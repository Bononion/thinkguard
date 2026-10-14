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