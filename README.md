# thinkguard-reproduction

## Setup

Needs a Linux host with an NVIDIA GPU. Run from a fresh shell:

```bash
git clone https://github.com/Bononion/thinkguard.git
cd thinkguard
export HF_HOME=/path/to/persistent/hf_cache       # model downloads
export CKPT_DIR=/path/to/persistent/checkpoints   # LoRA checkpoints
export HF_TOKEN=<your read token>                 # needed for gated models (Llama Guard 3-8B)
bash scripts/setup_env.sh                         # ends by printing versions and whether CUDA is available
source .venv/bin/activate
```

- Never commit the token; keep it in the shell environment only.
