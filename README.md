# thinkguard-reproduction

## Setup

Needs a Linux host with an NVIDIA GPU (vLLM ships Linux wheels only). Run from a fresh shell:

```bash
git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
export HF_HOME=/path/to/persistent/hf_cache       # model downloads
export CKPT_DIR=/path/to/persistent/checkpoints   # LoRA checkpoints
bash scripts/get_data.sh
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/smoke_vllm.py                      # on T4 GPUs: add --dtype half
```

Use a second env only if `uv pip install` reports a resolver conflict.
