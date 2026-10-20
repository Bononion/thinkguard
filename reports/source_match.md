# Source match

Critique rows are matched to BeaverTails on `pair_hash` (NFC, whitespace-collapsed prompt and response; `thinkguard/data/keys.py`). Report only: no labels are changed.

| Split | Rows | Distinct pairs | Pairs with >1 annotation | Pairs with conflicting is_safe | 15000 critique rows matched | Match |
|---|---|---|---|---|---|---|
| beavertails_30k_train | 27186 | 27117 | 68 | 3 | 15000 | 100.00% |
| beavertails_30k_test | 3021 | 3019 | 2 | 0 | 8 | 0.05% |
| beavertails_330k_train | 300567 | 99732 | 99732 | 27591 | 13526 | 90.17% |
| beavertails_330k_test | 33396 | 11088 | 11088 | 3075 | 1474 | 9.83% |

Source split: **beavertails_30k_train** (matches all 15000 rows). Other splits containing critique rows: beavertails_30k_test (8 rows), beavertails_330k_train (13526 rows), beavertails_330k_test (1474 rows).

Critique rows whose pair has conflicting is_safe annotations in beavertails_30k_train: 4 (rows 4774, 5475, 10042, 10912)

Critique rows not found in beavertails_30k_train: 0
