# Pilot

Measured throughput and memory for L and C on B (#61), and the overfit check before it (#14). All runs on Colab
A100-SXM4-40GB at `3b863db`, configs under `configs/train/`. Numbers are from each run's log; none is a reported
result.

## Overfit (#14)

32 smoke rows, 15 epochs (60 optimizer steps), scored on the same rows at the last step.

| | L (`20261112-1739-train-6a70264a`) | C (`20261112-1748-train-bbbc5c1c`) |
|---|---|---|
| Loss, last step | 0.001 | 0.007 |
| `eval_coverage` | 1.0 | 1.0 |
| `eval_unsafe_f1` | 1.0 | 0.944 |
| Decision and categories identical to the target | 32/32 | 30/32 |
| Peak allocated memory | not printed | 26.2 GiB |

C's critiques on smoke (`20261112-1754-eval-328fb0dc`, `configs/eval/smoke_C.yaml` with the best checkpoint): the
first three, read by eye, open as their targets do, with small wording changes ("identified as relevant" for
"triggered").

On a Colab L4 (22 GiB) both ran out of memory right after the first validation pass (step 4): L at microbatch 8
(`20261111-1649-train-6a70264a`) and C at microbatch 4 (`20261112-1649-train-879fc285`). Training needs the A100.
Those two passes had `eval_coverage` 0.0, with outputs such as `safe<|eot_id|>` (no `- Safety Assessment:` line yet)
while the loss was still 1.4-2.3.

## Pilot (#61)

1000 pilot rows (the 20 longest included), one epoch: 32 optimizer steps of 32 rows (microbatch 8, gradient
accumulation 4), then one validation pass over the 750 validation rows.

| | C (`20261112-1756-train-b6582e47`) | L (`20261112-1824-train-546501b1`) |
|---|---|---|
| Input tokens | 608,672 | 301,848 |
| Input tokens per row | 609 | 302 |
| Input tokens per optimizer step | 19,021 | 9,433 |
| Training time | 238.4 s | 121.9 s |
| Seconds per optimizer step | 7.45 | 3.81 |
| Tokens per second | 2,553 | 2,477 |
| One validation pass | 268.0 s | 220.4 s |
| Peak allocated memory | 31.3 GiB | 23.7 GiB |
| Loss, last step | 0.753 | 0.104 |
| `eval_unsafe_f1`, `eval_coverage` | 0.853, 1.0 | 0.827, 1.0 |

The full C run (`20261112-2147-train-a88bcf0c`) agrees: 2,544 tokens per second over its first 16 steps, 8,465,496
input tokens in its first 448 steps (1.01 epochs, about 590 per row), and validation passes of 216-250 s.

Not measured, against #61's task list:
- 32 optimizer steps instead of 100-200. The full C run's tokens per second match the pilot's, so more steps would
  not change the estimate.
- Supervised tokens per step and peak reserved memory: the trainer does not log them.
- LoRA weights changing: shown only by the loss falling and `eval_unsafe_f1` moving from step to step.
