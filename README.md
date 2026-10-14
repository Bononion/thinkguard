# thinkguard-reproduction

## Setup

Needs a Linux host with an NVIDIA GPU (vLLM ships Linux wheels only). Run from a fresh shell:

```bash
git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
export HF_HOME=/path/to/persistent/hf_cache       # model downloads
export CKPT_DIR=/path/to/persistent/checkpoints   # LoRA checkpoints
export HF_TOKEN=<your read token>                 # Llama Guard 3-8B is gated
bash scripts/get_data.sh
bash scripts/setup_env.sh
source .venv/bin/activate
python scripts/smoke_vllm.py                      # pre-Ampere GPUs (e.g. V100): add --dtype half
```

- `meta-llama/Llama-Guard-3-8B` is gated: request access on its Hugging Face page first. Never commit the token; keep it in the shell environment only.
- On a 16 GB GPU (e.g. Colab T4), the smoke test runs with --dtype half --cpu-offload-gb 6 --enforce-eager; training still needs >=24 GB.
- Use a second env only if `uv pip install` reports a resolver conflict.
