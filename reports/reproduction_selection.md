# Checkpoint selection for L and C (#63)

Validation `unsafe_f1` of every scored checkpoint, from `log_history` in each run's `checkpoint-1780/trainer_state.json`.
Rule (protocol.md, Checkpoint selection): the highest wins, a tie goes to the earlier step. Coverage is 1.000 at every
step of both runs. Bold = selected.

- L: run `20261113-1744-train-7565b8e7`, `checkpoint-896`.
- C: run `20261112-2147-train-a88bcf0c`, `checkpoint-1568`. It ties `checkpoint-1680` exactly (0.89241); the earlier
  step wins.

| step | epoch | L | C |
|---:|---:|---:|---:|
| 112 | 0.25 | 0.8495 | 0.8222 |
| 224 | 0.50 | 0.8818 | 0.8424 |
| 336 | 0.76 | 0.8937 | 0.8830 |
| 448 | 1.01 | 0.8891 | 0.8759 |
| 560 | 1.26 | 0.8898 | 0.8746 |
| 672 | 1.51 | 0.8598 | 0.8805 |
| 784 | 1.76 | 0.8791 | 0.8837 |
| 896 | 2.01 | **0.8983** | 0.8917 |
| 1008 | 2.27 | 0.8564 | 0.8826 |
| 1120 | 2.52 | 0.8879 | 0.8899 |
| 1232 | 2.77 | 0.8780 | 0.8923 |
| 1344 | 3.02 | 0.8919 | 0.8923 |
| 1456 | 3.27 | 0.8916 | 0.8876 |
| 1568 | 3.52 | 0.8899 | **0.8924** |
| 1680 | 3.78 | 0.8909 | 0.8924 |
| 1780 | 4.00 | 0.8917 | 0.8916 |
