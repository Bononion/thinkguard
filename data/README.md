# Data

## BeaverTails test

- The paper names no BeaverTails test split.
- The 15,000 training rows are 100% from `30k_train` (`scripts/check_source_split.py`).
- Test = `30k_test` (3,021 rows) minus 7 rows that overlap with training = 3,014 rows; unsafe count = 1,729.
- The authors' ThinkGuard.py loads 30k_test, consistent with this choice.