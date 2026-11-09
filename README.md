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
