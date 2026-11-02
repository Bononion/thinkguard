#!/usr/bin/env bash
# Sync .venv (Python 3.11) from uv.lock with the GPU stack. Linux + NVIDIA GPU only.
# Run from the repo root: bash scripts/setup_env.sh
set -euo pipefail

[ -f scripts/setup_env.sh ] || { echo "Run from the repo root: bash scripts/setup_env.sh" >&2; exit 1; }
for v in HF_HOME CKPT_DIR; do
  [ -n "${!v:-}" ] || { echo "$v is unset. Export it to a path on persistent storage first (see README Setup)." >&2; exit 1; }
done

if command -v uv >/dev/null; then
  UV=uv
else
  # --break-system-packages is only needed (and only known) on PEP 668 distros like Ubuntu 24.04
  python3 -m pip install --user uv || python3 -m pip install --user --break-system-packages uv
  UV="python3 -m uv"
fi

$UV sync --locked --group gpu
$UV run python -c "
import peft, torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('cuda available', torch.cuda.is_available())
"
