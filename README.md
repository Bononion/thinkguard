# thinkguard-reproduction

## Setup

Needs a Linux host with an NVIDIA GPU. Run from a fresh shell:

```bash
git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
export HF_HOME=/path/to/persistent/hf_cache       # model downloads
export CKPT_DIR=/path/to/persistent/checkpoints   # LoRA checkpoints
export HF_TOKEN=<your read token>                 # needed for gated models (Llama Guard 3-8B)
export THINKGUARD_RUNS=/path/to/persistent/runs   # run directories; on Colab, a Drive folder survives a disconnect
bash scripts/setup_env.sh                         # ends by printing versions and whether CUDA is available
```

- Run every command through `uv run` (e.g. `uv run pytest`); no need to activate `.venv`.
- Never commit the token; keep it in the shell environment only.

## Evaluate

```bash
uv run python -m thinkguard.data.normalize       # the splits read data/interim/, which is not in git
uv run python -m thinkguard.data.filter_train    # needed for train, validation and smoke
uv run python -m thinkguard.evaluate --config configs/eval/smoke_R.yaml
uv run python -m thinkguard.evaluate --config configs/eval/smoke_R.yaml --resume $THINKGUARD_RUNS/<id>  # after a disconnect
```

Predictions go to `$THINKGUARD_RUNS/<id>/predictions.jsonl`. `--resume` refuses if the code, config, packages or GPU
model differ from the interrupted run; then start over without it. Test splits run only from committed configs under
`configs/frozen/` (protocol.md, Test-set firewall).

`uv run python -m thinkguard.report scores --runs <id> [<id> ...]` scores stored predictions into
`results/scores.csv`, one row per run and metric, without a GPU or the model weights.

## Harness gate (#13)

R on the four test sets, from a fresh clone of a pushed commit (the firewall refuses uncommitted code). Each evaluate
prints its run directory; resume an interrupted one with `--resume $THINKGUARD_RUNS/<id>`.

```bash
for b in beavertails_30k_test toxicchat_0124_test openai_moderation wildguardtest; do
  uv run python -m thinkguard.evaluate --config configs/frozen/R_test_$b.yaml
done
uv run python -m thinkguard.report scores --runs <id1> <id2> <id3> <id4>
uv run python -m thinkguard.report gate          # results/gate.md; prints PASS or FAIL
```

Commit `results/scores.csv` and `results/gate.md`. On PASS, tag the commit that ran R `harness-frozen`; on FAIL,
stop: no training before the cause is found (protocol.md, Harness gate fixes).

## Train

```bash
uv run python -m thinkguard.training.data --mode train_critique --n 2   # the supervised tokens, by eye
uv run python -m thinkguard.train --config configs/train/overfit_critique.yaml   # #14, then overfit_label_category
uv run python -m thinkguard.evaluate --config configs/eval/smoke_C.yaml --adapter $THINKGUARD_RUNS/<id>/checkpoint-<n>
uv run python -m thinkguard.train --config configs/train/pilot_critique.yaml     # #61, then pilot_label_category
uv run python -m thinkguard.train --config configs/train/critique.yaml           # C; label_category.yaml for L
uv run python -m thinkguard.train --config configs/train/critique.yaml --resume $THINKGUARD_RUNS/<id>  # after a disconnect
```

Each run prints its directory. It is scored on validation 4 times per epoch (`validation/step-<n>.jsonl`, with
`eval_runtime` in the log), saves a `checkpoint-<n>/` each time, ends with `best_checkpoint.json` and prints its peak
GPU memory. `--resume` goes on from the last checkpoint with the same optimizer, scheduler, RNG and data position; it
refuses if the code, config, packages or GPU model differ. If the disconnect came while a checkpoint was being saved,
`--resume` fails on that half-written `checkpoint-<n>/`: delete it and `validation/step-<n>.jsonl` (scored before the
save, so a resume would reuse it instead of scoring the retrained step), and resume again from the one before (which
still has its optimizer state). `CKPT_DIR` must be set for `evaluate`, even with `--adapter`.
