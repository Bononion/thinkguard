# Compute plan

Status: proposal, to be approved by both members in #39. The roadmap's §13 cut order was not found, so the order
below is proposed here.

## Budget

- 100 Colab compute units (Bononion); the other member's units are not counted yet.
- Units per hour depend on the GPU and are shown by Colab when the runtime starts; each ledger row records the units
  its session used. Measured on Colab Pro: L4 about 1.94 units per hour for one session (2026-11-03), so 100 units is
  about 51 L4-hours. The first ledger row's 0.44 units include a second session that was open at the same time (3.09
  units per hour together). A100-SXM4-40GB: 6.73 units per hour, shown by Colab on 2026-11-12.
- The roadmap planned about 56 A100-hours for the whole project. 100 units is a small fraction of that, so the cut
  order below is expected to be used.

## Ledger

`reports/compute_ledger.csv` has one row per Colab session that runs anything from #34 on: `date`, `gpu`,
`units_before` and `units_after` (Colab's available units when the session opens and after it is disconnected, idle
time included) and `runs` (the run directory names, `thinkguard.runs`). Units used is `units_before - units_after`.
Run times come from each run's `manifest.json`. A session missing from the ledger shows up as a gap between one row's
`units_after` and the next row's `units_before`.

## Estimate

T_total = T_pilots + T_L + T_C + T_E + T_evaluation + T_benchmark + T_recovery

| Term | Estimate | Source |
|---|---|---|
| T_pilots | done, with the overfit runs | ledger |
| T_C | 5.2 A100-hours: 4 epochs of 445 steps at 8.4 s (the full run's rate), plus 16 validation passes of 4 min | `reports/pilot.md` |
| T_L | 2.9 A100-hours: 4 epochs of 445 steps at 3.8 s, plus 16 validation passes of 3.7 min | `reports/pilot.md` |
| T_E | TBD | #61, with the exit and layer-dropout costs from #36 |
| T_evaluation | about 1 A100-hour per condition on the four test sets (11,509 rows; validation scores 750 rows in about 4 min) | `reports/pilot.md` |
| T_benchmark | TBD | #89 |
| T_recovery | TBD | reruns after interruptions, from the ledger |

Against the budget, at 6.73 units per A100-hour: 81.2 units were left on 2026-11-12 after the pilots, before C
started. C, L, and B, L and C on the test sets need about 5.2 + 2.9 + 3 = 11.1 A100-hours, about 75 units, which
leaves about 6 units before any reruns. The reproduction comes first: T_E and T_benchmark are not budgeted until it
is done.

## Scope-cut order

A cut is made when the ledger and the estimate show that the remaining work exceeds the remaining budget. Cuts are
taken in this order, and each one is logged in the issue it removes, with the ledger numbers that justified it.

| Step | Cut | Saves |
|---|---|---|
| 1 | C-FT (#74) | one training run and its evaluation |
| 2 | Timing benchmark: drop W3, fewer requests and repeats (#89) | several A100-hours |
| 3 | Exit × draft grid: keep 1-2 configurations (#83, #84) | screening and timing runs |
| 4 | Extension scored on BeaverTails only; E-AR vs E-SSD token equality still checked on a small sample of the other three test sets (#85) | evaluation on three test sets |
| 5 | The whole extension (E-AR, E-SSD) | all of P6-P8 |
| 6 | Reproduction: fewer epochs or a shorter max length, each recorded as a deviation from the paper | depends on #61 |

C-FT comes first because it is the largest single saving and losing it only removes the C vs E attribution, a
limitation #27 already asks `protocol.md` to state. #74 calls it scope-cut step 3.

## Never cut

- Correctness checks and the tests that guard them
- Train-test leakage checks
- The L vs C comparison on all four test sets, and the R harness gate (#13)
